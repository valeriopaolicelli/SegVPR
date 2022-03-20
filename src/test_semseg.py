import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import os
import numpy as np

import logging
import datasets
import net.losses as loss
import net.functional as func
from metrics import StreamSegMetrics


def test(args, dataset, model):
    model.eval()
    metrics = StreamSegMetrics(args.num_classes)
    metrics.reset()
    test_loss = 0.0
    test_loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=4, pin_memory=True)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, ncols=80):
            output = model(images.cuda(), mode="only_classifier")
            output = func.interpol(x=output, size=(labels.shape[2], labels.shape[3]))
            loss_seg = loss.CrossEntropy2d(output, labels[:, 0, :, :].long().cuda())
            test_loss += loss_seg.item()
            _, output = output.max(dim=1)
            output = output.cpu().numpy()
            labels = labels.cpu().numpy()
            metrics.update(labels, output)

        logging.info(f"\tValidation loss on {dataset}: {test_loss/len(test_loader)}")
        score = metrics.get_results()
        logging.info(metrics.to_str_print(score))


def save_heatmap(args, eval_set, model, num_samples=50):
    test_dataloader = DataLoader(dataset=eval_set, num_workers=args.num_workers,
                                 batch_size=num_samples, pin_memory=True, shuffle=True)
    dataset_name = eval_set.info.split('/datasets/')[1].split('/')[0].split(':')[0]
    it_dataloader = iter(test_dataloader)
    model.eval()
    with torch.no_grad():
        logging.debug('Saving heatmap')
        images, _ = next(it_dataloader)
        i_size = images.size()
        att_output = model(images.cuda(), mode='att_scores')
        att_output = func.interpol(att_output, (i_size[2], i_size[3]))

        B, C, H, W = att_output.shape
        att_output = att_output.view(B, -1)
        att_output -= att_output.min(dim=1, keepdim=True)[0]
        att_output /= att_output.max(dim=1, keepdim=True)[0]
        att_output = att_output.view(B, C, H, W).cpu()

        for idx, (img, att) in tqdm(enumerate(zip(images, att_output))):
            img = datasets.inv_transform(img.detach().numpy())
            if not os.path.exists(os.path.join(args.output_folder, "heatmaps", dataset_name)):
                os.makedirs(os.path.join(args.output_folder, "heatmaps", dataset_name))
            cv2.imwrite(os.path.join(args.output_folder, "heatmaps", dataset_name, f"{idx}_att_image.jpg"), img)
            att = att.detach().numpy()
            mask = cv2.applyColorMap(np.uint8(255*att[0]), cv2.COLORMAP_JET)
            mask = mask * 0.5 + img * 0.5
            cv2.imwrite(os.path.join(args.output_folder, "heatmaps", dataset_name, f"{idx}_att_heatmap.jpg"), mask)

