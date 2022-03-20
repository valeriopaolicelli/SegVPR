import logging
from datetime import datetime
from os import remove
from os.path import join
from glob import glob
import torch

import utils.parser as parser
import utils.util as util
import utils.commons as commons
import datasets
import test
import train
import test_semseg
from net.scheduler import PolyLR

# ######################################## SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"runs/{args.exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.pause_while_running(args.wait)
args = commons.set_paths_ws(args)
commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

# ######################################## DATASETS #########################################

query_train_set = datasets.QueryDataset(dataset_root=args.dataset_root,
                                        gallery_path=args.train_g, query_path=args.train_q,
                                        output_folder=args.output_folder, n_neg=args.n_neg,
                                        train_pos_dist_threshold=args.train_pos_dist_threshold,
                                        val_pos_dist_threshold=args.val_pos_dist_threshold)
logging.info(f"Train query set: {query_train_set}")

whole_train_set = datasets.WholeDataset(args.dataset_root, args.train_g, args.train_q,
                                        val_pos_dist_threshold=args.val_pos_dist_threshold)
logging.info(f"Train whole set: {whole_train_set}")

DA_dataset = datasets.DADataset(args.DA_datasets)
logging.info(f"{DA_dataset}")

whole_val_set = datasets.WholeDataset(args.dataset_root_val, args.val_g, args.val_q,
                                      val_pos_dist_threshold=args.val_pos_dist_threshold)
logging.info(f"Val set left/right: {whole_val_set}")

whole_test_set = []
for gallery, queries in zip(args.test_g.split('+'), args.test_q.split('+')):
    whole_test_set.append(datasets.WholeDataset(args.dataset_root_test, gallery, queries,
                                                val_pos_dist_threshold=args.val_pos_dist_threshold))
    logging.info(f"Test set: {whole_test_set[-1]}")

whole_sem_test_set = datasets.SemDataset(join(args.dataset_root_val, args.val_g))
logging.info(f"Sem test set: {whole_sem_test_set}")

# ######################################## MODEL #########################################
model = util.build_model(args, whole_train_set)

# ######################################## OPTIMIZER #####################################
params = [{'params': model.sem_branch.encoder.parameters(), 'lr': args.lr * args.batch_size * 2},
          {'params': model.sem_branch.cls.parameters(), 'lr': 10 * args.lr * args.batch_size * 2},
          {'params': model.attention_layers.parameters(), 'lr': args.lr * args.batch_size * 2}]

if args.semnet == 'pspnet':
    params.append({'params': model.sem_branch.ppm.parameters(), 'lr': 10 * args.lr * args.batch_size * 2})

optimizer = torch.optim.SGD(params,
                            lr=args.lr * args.batch_size * 2,
                            momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = PolyLR(optimizer, max_iters=250000 / args.batch_size, power=0.9)

optimizer_d = torch.optim.Adam(model.DA.parameters(), lr=args.d_lr * args.batch_size * 2, betas=(0.9, 0.99))
scheduler_d = PolyLR(optimizer_d,
                     max_iters=250000 / args.batch_size, power=0.9)

not_improved = 0
best_score = 0
# ######################################## RESUME #########################################
if args.resume:
    model, optimizer, scheduler, optimizer_d, scheduler_d, \
        best_score = util.resume_train(args, model, optimizer, scheduler, optimizer_d, scheduler_d)

# ######################################## TRAINING #########################################
for epoch in range(0, args.n_epochs):
    logging.info(f"Start training epoch: {epoch + 1:02d}")

    train.train(args, epoch, model, optimizer, scheduler, optimizer_d, scheduler_d,
                whole_train_set, query_train_set, DA_dataset)

    logging.info(f"VPR validation su val set")
    recalls, recalls_str_val = test.test(args, whole_val_set, model)
    logging.info(f"\tRecalls on val set {whole_val_set}: {recalls_str_val}")

    logging.info(f"VPR validation su test sets")
    recalls_str_test = []
    for test_set in whole_test_set:
        recalls_str_test.append(test.test(args, test_set, model)[1])
        logging.info(f"\tRecalls on test set {test_set}: {recalls_str_test[-1]}")

    logging.info(f"Semseg validation")
    test_semseg.test(args, whole_sem_test_set, model)

    if recalls[1] > best_score:
        logging.info(
            f"\tImproved: previous best recall@5 = {best_score * 100:.1f}, current recall@5 = {recalls[5] * 100:.1f}")
        is_best = True
        best_score = recalls[1]
        not_improved = 0
    else:
        is_best = False
        if not_improved >= args.patience:
            logging.info(f"\tPerformance did not improve for {not_improved} epochs. Stop training.")
            break
        not_improved += 1
        logging.info(
            f"\tNot improved: {not_improved} / {args.patience}: best recall@5 = {best_score * 100:.1f}, current "
            f"recall@5 = {recalls[5] * 100:.1f}")

    util.save_checkpoint(args=args, state={"state_dict": model.state_dict(),
                                           "recalls": recalls, "best_score": best_score,
                                           "optimizer": optimizer.state_dict(),
                                           "scheduler": scheduler.state_dict(),
                                           "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
                                           "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
                                           }, is_best=is_best)

logging.info(f"Best recall@5: {best_score * 100:.1f}. Trained for {str(datetime.now() - start_time)[:-7]}")

for p in glob(f'{args.output_folder}/*.hdf5'):
    remove(p)

# ######################################## TEST on TEST SET #########################################
best_model_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_dict["state_dict"])

logging.info(f"VPR validation su val set")
recalls, recalls_str_val = test.test(args, whole_val_set, model)
logging.info(f"\tRecalls on val set {whole_val_set}: {recalls_str_val}")

logging.info(f"VPR validation su test sets")
recalls_str_test = []
for test_set in whole_test_set:
    recalls_str_test.append(test.test(args, test_set, model)[1])
    logging.info(f"\tRecalls on test set {test_set}: {recalls_str_test[-1]}")

logging.info(f"Semseg validation")
test_semseg.test(args, whole_sem_test_set, model)
