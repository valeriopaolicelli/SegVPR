from datetime import datetime
from os.path import join
import torch

import logging
import utils.parser as parser
import utils.util as util
import utils.commons as commons
import datasets
import test
import test_semseg


######################################### SETUP #########################################
args = parser.parse_arguments()
args.output_folder = f"runs/{args.exp_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
commons.setup_logging(args.output_folder)
args = commons.set_paths_ws(args)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

assert args.resume is not None, "resume is set to None, please set it to the path of the checkpoint to resume"

######################################### MODEL #########################################
model = util.build_model(args)

######################################### RESUME #########################################
model_state_dict = torch.load(args.resume)["state_dict"]
model.load_state_dict(model_state_dict, strict=False)

######################################### DATASETS #########################################
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

######################################### TEST on VAL SET #########################################
logging.info(f"VPR validation su val set")
recalls, recalls_str_val = test.test(args, whole_val_set, model)
logging.info(f"\tRecalls on val set {whole_val_set}: {recalls_str_val}")

######################################### TEST on TEST SET #########################################
logging.info(f"VPR validation su test sets")
recalls_str_test = []
for test_set in whole_test_set:
    recalls_str_test.append(test.test(args, test_set, model)[1])
    logging.info(f"\tRecalls on test set {test_set}: {recalls_str_test[-1]}")

logging.info(f"Semseg validation")
test_semseg.test(args, whole_sem_test_set, model)

