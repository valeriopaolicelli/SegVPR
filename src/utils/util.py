
import logging
from math import ceil
from os.path import join, abspath, exists

from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from tqdm import tqdm

from net import network, resnet_vpr, pooling, discriminator, get_semnet

import json


def save_checkpoint(args, state, is_best):
    if is_best:
        torch.save(state, f"{args.output_folder}/best_model.pth")


def remove_module_from_keys(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '') if 'module.' in k else k
        new_dict[k] = v
    return new_dict


def resume_train(args, model, optimizer, scheduler, optimizer_d, scheduler_d):
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    state_dict = remove_module_from_keys(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    best_score = checkpoint["best_score"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if ("optimizer_d" in checkpoint) and ("scheduler_d" in checkpoint):
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
    logging.debug(f"Loaded checkpoint: current_best_recall@5 = {best_score}")
    return model, optimizer, scheduler, optimizer_d, scheduler_d, best_score


def build_model(args, cluster_set=None):
    logging.debug(f"Building SemVPR: {args.arch.upper()} ({args.pretrain.upper()}) + ATT + "
                  f"L2 GeM (from {args.scale_indexes}) + {args.semnet.upper()} with ATT applied on sem output + "
                  f"DCGAN")

    encoder, args.encoder_dim, args.pool_feat_dim = getattr(resnet_vpr, args.arch)(pretrained=True,
                                                                                   pretrain=args.pretrain,
                                                                                   path_model=args.path_model,
                                                                                   scale_indexes=args.scale_indexes)

    if args.train_enc_from != 'all':
        logging.debug(f"Train from encoder layer{args.train_enc_from}, freeze the previous ones")
        for name, child in encoder.named_children():
            if name == f"layer{args.train_enc_from}":
                break
            for name2, params in child.named_parameters():
                params.requires_grad = False
    else:
        logging.debug(f"Train all layers")

    sem_branch = getattr(get_semnet, f'get_{args.semnet.lower()}')(encoder=encoder, encoder_dim=args.encoder_dim,
                                                                   classes=args.num_classes).to(args.device)

    vpr_pooling = getattr(pooling, args.pooling.upper())(encoder_dim=args.pool_feat_dim,
                                                         num_clusters=args.num_clusters)

    DA_discr = getattr(discriminator, f'get_{args.DA_type.lower()}')(num_classes=args.num_classes)

    model = network.SemVPR(vpr_pooling=vpr_pooling, sem_branch=sem_branch,
                           num_queries=args.batch_size, num_negatives=args.n_neg, DA=DA_discr)

    return model.to(args.device)


classes_name = {
            0: "Road",
            1: "Sidewalk",
            2: "Building",
            3: "Wall",
            4: "Fence",
            5: "Pole",
            6: "TLight",
            7: "TSign",
            8: "Vegetation",
            9: "Terrain",
            10: "Sky",
            11: "Person",
            12: "Rider",
            13: "Vehicle",
            14: "Truck",
            15: "Motorcycle",
            16: "Bicycle"
        }

scores_classes_hardcoded = [5, 20, 40, 30, 20, 30, 30, 30, 5] + 8*[0]


def format_str_class_scores(class_scores):
    i, avg_scores = 0, '\n'
    while i < len(class_scores):
        avg_scores += f'{classes_name[i]}: {class_scores[i]:.4f} ' + (10 - len(classes_name[i])) * ' '
        if i + 1 < len(class_scores):
            avg_scores += f'| {classes_name[i + 1]}: {class_scores[i + 1]:.4f} ' + (10 - len(classes_name[i + 1])) * ' '
            if i + 2 < len(class_scores):
                avg_scores += f'| {classes_name[i + 2]}: {class_scores[i + 2]:.4f} ' \
                              + (10 - len(classes_name[i + 2])) * ' '
                if i + 3 < len(class_scores):
                    avg_scores += f'| {classes_name[i + 3]}: {class_scores[i + 3]:.4f} ' \
                                  + (10 - len(classes_name[i + 3])) * ' '
                    if i + 4 < len(class_scores):
                        avg_scores += f'| {classes_name[i + 4]}: {class_scores[i + 4]:.4f} ' \
                                      + (10 - len(classes_name[i + 4])) * ' '
                        if i + 5 < len(class_scores):
                            avg_scores += f'| {classes_name[i + 5]}: {class_scores[i + 5]:.4f} ' \
                                          + (10 - len(classes_name[i + 5])) * ' '
                            if i + 6 < len(class_scores):
                                avg_scores += f'| {classes_name[i + 6]}: {class_scores[i + 6]:.4f} ' \
                                              + (10 - len(classes_name[i + 6])) * ' '
                                if i + 7 == len(class_scores) - 1:
                                    avg_scores += f'| {classes_name[i + 7]}: {class_scores[i + 7]:.4f}\n'
                                else:
                                    avg_scores += '\n'
        i += 7
    return avg_scores


def get_target_tensor(input_tensor, mode):
    # Source tensor = 0.0
    # Target tensor =  1.0
    domain_class = 0.0 if mode == 'source' else 1.0
    tensor = torch.FloatTensor(1).fill_(domain_class)
    tensor = tensor.expand_as(input_tensor)
    return tensor


