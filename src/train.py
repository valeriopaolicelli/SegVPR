import h5py
import logging
import math
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

import datasets
import net.losses as loss


def build_cache(args, dataset, query_set, cache_name, model):
    num_queries = len(query_set)
    queries_sample = np.random.choice(np.arange(num_queries), args.cache_refresh_rate, replace=False)
    num_galleries = dataset.db_struct.num_gallery
    gallery_indexes = list(range(num_galleries))
    subset = Subset(dataset, gallery_indexes + list(queries_sample + num_galleries))
    subset_dl = DataLoader(dataset=subset, num_workers=args.num_workers,
                           batch_size=args.cache_batch_size, shuffle=False,
                           pin_memory=True)
    model.eval()
    cache = np.zeros((len(dataset), args.pool_feat_dim), dtype=np.float32)
    with torch.no_grad():
        for inputs, indices in tqdm(subset_dl, ncols=100):
            inputs = inputs.to(args.device)
            out = model(inputs, mode="only_embeddings")
            cache[indices.detach().numpy(), :] = out.view(inputs.shape[0], -1).detach().cpu().numpy()
    with h5py.File(f"{args.output_folder}/{cache_name}.hdf5", mode='w') as h5:
        h5.create_dataset("cache", data=cache, dtype=np.float32)
    del inputs, out, cache
    sub_query_train_set = Subset(dataset=query_set, indices=queries_sample)
    query_dataloader = DataLoader(dataset=sub_query_train_set, num_workers=args.num_workers,
                                  batch_size=args.batch_size, shuffle=False, drop_last=True,
                                  collate_fn=datasets.collate_fn, pin_memory=True)
    return query_dataloader


def train(args, epoch, model, optimizer, scheduler, optimizer_d, scheduler_d,
          whole_train_set, query_train_set, DA_dataset):
    epoch_start_time = datetime.now()
    epoch_losses_vpr = np.zeros((0, 1), dtype=np.float32)
    epoch_losses_sem = np.zeros((0, 1), dtype=np.float32)

    DA_dataloader = DataLoader(dataset=DA_dataset, num_workers=args.num_workers,
                               batch_size=(args.batch_size * 2 + args.n_neg * args.batch_size),
                               shuffle=True, pin_memory=True, drop_last=True)
    DA_dataloader_it = iter(DA_dataloader)
    epoch_losses_adv = np.zeros((0, 1), dtype=np.float32)
    epoch_losses_DA = np.zeros((0, 1), dtype=np.float32)

    assert args.queries_per_epoch % args.cache_refresh_rate == 0
    num_train_loops = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for index_train_loop in range(num_train_loops):
        logging.debug(f"[{index_train_loop + 1}/{num_train_loops}] Build cache")
        query_dataloader = build_cache(args, whole_train_set, query_train_set,
                                       cache_name="cache", model=model)

        model.train()
        for query, positives, negatives, neg_counts, query_label, positive_label, negatives_label \
                in tqdm(query_dataloader, ncols=100):
            if query is None:
                continue  # in case we get an empty batch

            try:
                target_images = next(DA_dataloader_it)
            except:
                DA_dataloader_it = iter(DA_dataloader)
                target_images = next(DA_dataloader_it)

            target_images = target_images.cuda()

            B, C, H, W = query.shape
            n_neg = torch.sum(neg_counts)
            inputs = torch.cat([query, positives, negatives])
            inputs = inputs.to(device=args.device)
            labels = torch.cat([query_label, positive_label, negatives_label])
            labels = labels[:, 0, :, :].long().cuda()

            multiscale_feat, final_sem_source = model(inputs, mode='full_net')  # already weighted by the attention
            global_encoding = model.concat_feat(multiscale_feat)
            global_q, global_p, global_n = torch.split(global_encoding.view(inputs.shape[0], -1), [B, B, n_neg])

            optimizer.zero_grad()
            optimizer_d.zero_grad()

            total_loss = loss.TripletMarginLoss(global_q, global_p, global_n, neg_counts, n_neg)
            epoch_losses_vpr = np.append(epoch_losses_vpr, total_loss.item())

            loss_ce_sem = loss.CrossEntropy2d(final_sem_source, labels, weight=None, reduction='mean')
            epoch_losses_sem = np.append(epoch_losses_sem, loss_ce_sem.item())
            total_loss += args.sem_loss_weight * loss_ce_sem

            final_sem_target = model(target_images, mode='only_classifier')
            for param in model.DA.parameters():
                param.requires_grad = False
            loss_adv = model.DA(inputs=F.softmax(final_sem_target, dim=1), class_label='source')
            total_loss += (args.sem_loss_weight * loss_adv * args.adv_loss_weight)
            epoch_losses_adv = np.append(epoch_losses_adv, loss_adv.item())

            # Train the entire network: Triplet loss + Sem Loss source + Sem Loss target + Adv Loss
            total_loss.backward()

            for param in model.DA.parameters():
                param.requires_grad = True

            final_sem_source = final_sem_source.detach()  # train only the discriminator
            final_sem_target = final_sem_target.detach()  # train only the discriminator
            source_d_loss = model.DA(inputs=F.softmax(final_sem_source, dim=1), class_label='source') / 2
            (args.da_loss_weight * source_d_loss).backward()
            target_d_loss = model.DA(inputs=F.softmax(final_sem_target, dim=1), class_label='target') / 2
            (args.da_loss_weight * target_d_loss).backward()
            epoch_losses_DA = np.append(epoch_losses_DA, (source_d_loss.item() + target_d_loss.item()))

            optimizer_d.step()
            scheduler_d.step()
            optimizer.step()
            scheduler.step()

        avg_ep_sem_loss = f'{epoch_losses_sem.mean():.4f}'
        avg_ep_adv_loss = f'{epoch_losses_adv.mean():.4f}'
        avg_ep_DA_loss = f'{epoch_losses_DA.mean():.4f}'

        logging.debug(f"Epoch[{epoch + 1:02d}]({index_train_loop + 1}/{num_train_loops}): " +
                      f"avg_ep_vpr_loss= {epoch_losses_vpr.mean():.4f} " +
                      f"| avg_ep_sem_loss= {avg_ep_sem_loss} " +
                      f"| avg_ep_adv_loss= {avg_ep_adv_loss} " +
                      f"| avg_ep_DA_loss= {avg_ep_DA_loss}")

    avg_ep_sem_loss = f'{epoch_losses_sem.mean():.4f}'
    avg_ep_adv_loss = f'{epoch_losses_adv.mean():.4f}'
    avg_ep_DA_loss = f'{epoch_losses_DA.mean():.4f}'

    logging.info(f"Finished epoch {epoch + 1:02d} in {str(datetime.now() - epoch_start_time)[:-7]}: " +
                 f"avg_ep_vpr_loss= {epoch_losses_vpr.mean():.4f} " +
                 f"| avg_ep_sem_loss= {avg_ep_sem_loss} " +
                 f"| avg_ep_adv_loss= {avg_ep_adv_loss} "
                 f"| avg_ep_DA_loss= {avg_ep_DA_loss}")
