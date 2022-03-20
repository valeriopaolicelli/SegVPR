import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="SemVPR", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Network
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--semnet", type=str, default="deeplab", choices=["pspnet", "deeplab"])
    parser.add_argument("--num_classes", type=int, default=17, help="Number of semantic classes")
    parser.add_argument("--pretrain", type=str, default="imagenet", choices=["imagenet"])
    parser.add_argument("--pooling", type=str, default="gem", choices=["gem"])
    parser.add_argument("--scale_indexes", type=int, nargs='+', default=[3, 4],
                        help='Which conv blocks use for the multi-scale pooling (index goes from 1 to 4)')
    parser.add_argument("--train_enc_from", type=str, default="all", choices=["all", "1", "2", "3", "4"])
    parser.add_argument("--DA_type", type=str, default="dcgan", choices=["dcgan"],
                        help="Use DA for synthetic/real world domain adaptation")

    # Train params
    parser.add_argument("--batch_size", type=int, default=2, help="Number of triplets (query, pos, negs).")
    parser.add_argument("--cache_batch_size", type=int, default=24, help="Batch size for inference.")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000, help="How often to refresh cache.")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--num_clusters", type=int, default=64, help="Number of clusters for NetVLAD layer")
    parser.add_argument("--n_neg", type=int, default=1, help="Number of negatives per query")
    parser.add_argument("--train_pos_dist_threshold", type=int, default=3, help="Train positive distance")
    parser.add_argument("--val_pos_dist_threshold", type=int, default=5, help="Val positive distance")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for the triplet loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning Rate.")
    parser.add_argument("--d_lr", type=float, default=1e-4, help="Discriminator learning Rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optim momentum")
    parser.add_argument("--weight_decay", type=float, default=0.005, help="Optim weight decay")
    parser.add_argument("--resume", type=str, default=None, help="Path to load checkpoint from.")
    parser.add_argument('--adv_loss_weight', type=float, default=0.001, help='Weight adv loss')
    parser.add_argument('--sem_loss_weight', type=float, default=0.5, help='Weight semseg loss')
    parser.add_argument('--da_loss_weight', type=float, default=0.5, help='Weight discr loss')

    # Exp params
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=3, help="num_workers for all dataloaders")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")
    parser.add_argument("--results_table", type=str, default="./results.json",
                        help="JSON file with all runs results")
    parser.add_argument('--wait', type=int, default=-1, help='PID of process to wait before start.')

    # Datasets
    parser.add_argument("--dataset_root", type=str, default="/datasets/idda/images",
                        help="Root path of IDDAv2 train dataset")
    parser.add_argument("--train_g", type=str, default="town10/gallery/front_rear", help="Path train gallery")
    parser.add_argument("--train_q", type=str, default="town10/queries_rain/front_rear", help="Path train query")

    parser.add_argument("--dataset_root_val", type=str, default="/datasets/idda/images",
                        help="Root path of IDDAv2 val dataset")
    parser.add_argument("--val_g", type=str, default="town3/gallery/left_right", help="Path val gallery")
    parser.add_argument("--val_q", type=str, default="town3/queries_rain/left_right", help="Path val query")

    parser.add_argument("--dataset_root_test", type=str, default="/datasets/robotcar",
                        help="Root path of test dataset")
    parser.add_argument("--test_g", type=str, default="overcast+overcast+overcast+overcast", help="Path val gallery")
    parser.add_argument("--test_q", type=str, default="rain+snow+sun+night", help="Path val query")

    parser.add_argument('--DA_datasets', type=str, default='/datasets/robotcar/all/',
                        help='Paths real domain dataset')

    return parser.parse_args()

