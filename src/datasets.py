import random

import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple
from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
import faiss
from glob import glob
import os
from os.path import join, exists


def inv_transform(image, mean=(0.485, 0.456, 0.406)):
    image = np.transpose(image, (1, 2, 0))
    image += mean  # normalize
    return image


def idda_cs_mapping(mask):
    mapping = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 16, 15, 12, 9, 10, 255, 255, 255, 255, 255, 14]
    mask_copy = mask.copy()
    for c in range(len(mapping)):
        mask_copy[mask == c] = mapping[c]
    return mask_copy


def get_crop_dim(image_shape, scale_percent=60):
    width = int(image_shape[0] * scale_percent / 100)
    height = int(image_shape[1] * scale_percent / 100)
    return width, height


def get_random_crop(images, labels, i_size, dim):
    max_h = i_size[1] - dim[1]
    max_w = i_size[0] - dim[0]
    h = np.random.randint(0, max_h)
    w = np.random.randint(0, max_w)
    for idx, (image, label) in enumerate(zip(images, labels)):
        images[idx] = image[h: h + dim[1], w: w + dim[0]]
        labels[idx] = label[h: h + dim[1], w: w + dim[0]] if label is not None else None
    return images, labels


def random_horizontal_flip(images, labels):
    flip_prob = random.randint(0, 1)
    if flip_prob == 1:
        for idx, (image, label) in enumerate(zip(images, labels)):
            images[idx] = cv2.flip(image, 1)
            labels[idx] = cv2.flip(label, 1) if label is not None else None
    return images, labels


def transform(images_paths, labels_paths, mean=(0.485, 0.456, 0.406), data_aug=True, resize=False, new_dim=None):
    images, labels = [], []
    for idx, (image, label) in enumerate(zip(images_paths, labels_paths)):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        label = cv2.imread(label) if label is not None else None
        labels.append(label)

    i_size = (images[0].shape[1], images[0].shape[0])
    if data_aug:
        crop_dim = get_crop_dim(image_shape=i_size, scale_percent=80)
        images, labels = get_random_crop(images, labels, i_size=i_size, dim=crop_dim)  # random crop
        images, labels = random_horizontal_flip(images, labels)

    if resize:
        new_dim = get_crop_dim(image_shape=i_size, scale_percent=50) if new_dim is None else new_dim
        for idx, (image, label) in enumerate(zip(images, labels)):
            images[idx] = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
            labels[idx] = cv2.resize(label, new_dim, interpolation=cv2.INTER_NEAREST) if label is not None else None

    for idx, (image, label) in enumerate(zip(images, labels)):
        image = np.float32(image)
        image = image[:, :, ::-1]  # change to BGR
        image -= mean  # normalize
        images[idx] = transforms.ToTensor()(image.copy())
        if label is not None:
            label = np.asarray(label, np.float32)[:, :, 2]
            label = idda_cs_mapping(label)
            labels[idx] = transforms.ToTensor()(label.copy())
    return images, labels


def sem_transform(image_path, label_path, mean=(0.485, 0.456, 0.406)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)
    image = image[:, :, ::-1]  # change to BGR
    image -= mean  # normalize
    image = transforms.ToTensor()(image.copy())
    label = cv2.imread(label_path) if label_path is not None else label_path
    label = np.asarray(label, np.float32)[:, :, 2]
    label = idda_cs_mapping(label)
    label = transforms.ToTensor()(label.copy())
    return image, label


def filter_per_orientation(q_images, db_images, positives_per_query):
    filtered_positives = []
    for query_idx, positives in enumerate(positives_per_query):
        q_positives = []
        for p_idx in positives:
            if '/front/' in q_images[query_idx] and '/front/' in db_images[p_idx]:
                q_positives.append(p_idx)
            elif '/rear/' in q_images[query_idx] and '/rear/' in db_images[p_idx]:
                q_positives.append(p_idx)
            elif '/left/' in q_images[query_idx] and '/left/' in db_images[p_idx]:
                q_positives.append(p_idx)
            elif '/right/' in q_images[query_idx] and '/right/' in db_images[p_idx]:
                q_positives.append(p_idx)
        filtered_positives.append(np.array(q_positives))
    return filtered_positives


db_struct = namedtuple("db_struct",
                       ["gallery_images", "gallery_utms", "query_images", "query_utms", "num_gallery", "num_queries"])


def parse_db_struct(dataset_root, gallery_path, query_path):
    gallery_path = join(dataset_root, gallery_path)
    query_path = join(dataset_root, query_path)
    if not os.path.exists(gallery_path): raise Exception(f"{gallery_path} does not exist")
    if not os.path.exists(query_path): raise Exception(f"{query_path} does not exist")
    db_images = sorted(glob(f"{gallery_path}/**/*.jpg", recursive=True))
    q_images = sorted(glob(f"{query_path}/**/*.jpg", recursive=True))
    db_utms = np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in db_images])
    q_utms = np.array([(float(img.split("@")[1]), float(img.split("@")[2])) for img in q_images])
    num_gallery = len(db_images)
    num_queries = len(q_images)
    return db_struct(db_images, db_utms, q_images, q_utms, num_gallery, num_queries)


class WholeDataset(data.Dataset):
    # Dataset with both gallery and query images, used for inference (testing and building cache)
    def __init__(self, dataset_root, gallery_path, query_path, val_pos_dist_threshold=25):
        super().__init__()
        self.val_pos_dist_threshold = val_pos_dist_threshold if '/idda/' in dataset_root else 25
        self.db_struct = parse_db_struct(dataset_root, gallery_path, query_path)
        self.images = [dbIm for dbIm in self.db_struct.gallery_images]
        self.images += [qIm for qIm in self.db_struct.query_images]
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct.gallery_utms)
        self.positives_per_query = knn.radius_neighbors(self.db_struct.query_utms,
                                                        radius=self.val_pos_dist_threshold, return_distance=False)
        if '/idda/' in dataset_root:
            self.positives_per_query = filter_per_orientation(self.db_struct.query_images, self.db_struct.gallery_images,
                                                              self.positives_per_query)

        self.info = f"< WholeDataset {dataset_root}: queries: {query_path} ({self.db_struct.num_queries}); " \
                    f"gallery: {gallery_path} ({self.db_struct.num_gallery}) >"

    def __getitem__(self, index):
        resize = True if ('/idda/' in self.info) or ('/a2d2/' in self.info) or ('/mapillary/' in self.info) \
                         or ('/bdd100k/' in self.info) else False
        img = transform(images_paths=[self.images[index]], labels_paths=[None], data_aug=False, resize=resize)[0][0]
        return img, index

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return self.info

    def getPositives(self):
        return self.positives_per_query


def collate_fn(batch):
    batch = list(batch)
    if len(batch) == 0:
        return None, None, None, None, None, None, None
    query, positive, negatives, query_label, positive_label, negatives_label = zip(*batch)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    neg_counts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    query_label = data.dataloader.default_collate(query_label)
    positive_label = data.dataloader.default_collate(positive_label)
    negatives_label = torch.cat(negatives_label, 0)
    return query, positive, negatives, neg_counts, query_label, positive_label, negatives_label


class QueryDataset(data.Dataset):
    def __init__(self, dataset_root, gallery_path, query_path, output_folder,
                 train_pos_dist_threshold=10, val_pos_dist_threshold=25, n_neg=5, cache_name='cache'):
        super().__init__()
        self.output_folder = output_folder
        self.cache_name = cache_name
        self.train_pos_dist_threshold = train_pos_dist_threshold if '/idda/' in dataset_root else 10
        self.val_pos_dist_threshold = val_pos_dist_threshold if '/idda/' in dataset_root else 25
        self.margin = 0.1
        self.db_struct = parse_db_struct(dataset_root, gallery_path, query_path)
        self.n_neg_samples = 1000  # Number of negatives to randomly sample
        self.n_neg = n_neg  # Number of negatives per query in each batch
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.db_struct.gallery_utms)
        self.positives_per_query = list(knn.radius_neighbors(self.db_struct.query_utms,
                                                             radius=self.train_pos_dist_threshold,
                                                             return_distance=False))
        self.positives_per_query = [np.sort(positives) for positives in self.positives_per_query]
        if '/idda/' in dataset_root:
            self.positives_per_query = filter_per_orientation(self.db_struct.query_images, self.db_struct.gallery_images,
                                                              self.positives_per_query)

        # Make sure that each query in the train set has at least one positive, otherwise it should be deleted
        queries_without_any_positive = np.where(np.array([len(p) for p in self.positives_per_query]) == 0)[0]
        if len(queries_without_any_positive) > 0:
            with open(f"{output_folder}/queries_without_any_positive.txt", "w") as file:
                for query_index in queries_without_any_positive:
                    file.write(f"{self.db_struct.query_images[query_index]}\n")
            raise Exception(f"There are {len(queries_without_any_positive)} queries in the training " +
                            f"set without any positive (within {self.train_pos_dist_threshold} meters) in " +
                            "the gallery! Please remove these images, as they're not used for training. " +
                            "The paths of these images have been saved in " +
                            f"{output_folder}/queries_without_any_positive.txt")

        positives = knn.radius_neighbors(self.db_struct.query_utms,
                                         radius=25, return_distance=False)
        self.negatives_per_query = []
        for pos in positives:
            self.negatives_per_query.append(np.setdiff1d(np.arange(self.db_struct.num_gallery),
                                                         pos, assume_unique=True))
        self.neg_cache = [np.empty((0,)) for _ in range(self.db_struct.num_queries)]
        self.info = f"< QueryDataset {dataset_root}: queries: {query_path} ({self.db_struct.num_queries}); " \
                    f"gallery: {gallery_path} ({self.db_struct.num_gallery}) >"

    def __getitem__(self, index):
        with h5py.File(f"{self.output_folder}/{self.cache_name}.hdf5", mode="r") as h5:
            cache = h5.get("cache")
            features_dim = cache.shape[1]
            queries_offset = self.db_struct.num_gallery
            query_features = cache[index + queries_offset]
            if np.all(query_features == 0):
                raise Exception(
                    f"For query {self.db_struct.query_images[index]} with index {index} "
                    f"features have not been computed!!!")
            positives_features = cache[self.positives_per_query[index].tolist()]
            faiss_index = faiss.IndexFlatL2(features_dim)
            faiss_index.add(positives_features)
            # Search the best positive (within 10 meters AND nearest in features space)
            best_pos_dist, best_pos_num = faiss_index.search(query_features.reshape(1, -1), 1)
            best_pos_index = self.positives_per_query[index][best_pos_num[0]].item()
            # Sample 1000 negatives randomly and concatenate them with the previous top 10 negatives (neg_cache)
            neg_samples = np.random.choice(self.negatives_per_query[index], self.n_neg_samples, replace=False)
            neg_samples = np.unique(np.concatenate([self.neg_cache[index], neg_samples]))
            neg_features = np.array([cache[int(neg_sample)] for neg_sample in neg_samples])

        faiss_index = faiss.IndexFlatL2(features_dim)
        faiss_index.add(neg_features)
        # Search the nearest negatives (further than val_pos_dist_threshold meters and nearest in features space)
        neg_dist, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.n_neg)
        neg_nums = neg_nums.reshape(-1)
        neg_indices = neg_samples[neg_nums].astype(np.int32)
        self.neg_cache[index] = neg_indices  # Update nearest negatives in neg_cache

        query_path = self.db_struct.query_images[index]
        query_label_path = query_path.replace('/queries', '/queries_labels')
        query_label_path = query_label_path.replace('.jpg', '.png')
        if not os.path.exists(query_label_path): raise Exception(f"Query label: {query_label_path} does not exist")

        positive_path = self.db_struct.gallery_images[best_pos_index]
        positive_label_path = positive_path.replace('/gallery/', '/gallery_labels/')
        positive_label_path = positive_label_path.replace('.jpg', '.png')
        if not os.path.exists(positive_label_path): raise Exception(f"Pos label: {positive_label_path} does not exist")

        images_paths = [query_path, positive_path]
        labels_paths = [query_label_path, positive_label_path]

        for i in neg_indices:
            negative_path = self.db_struct.gallery_images[i]
            negative_label_path = negative_path.replace('/gallery/', '/gallery_labels/')
            negative_label_path = negative_label_path.replace('.jpg', '.png')
            images_paths.append(negative_path)
            labels_paths.append(negative_label_path)

        images, labels = transform(images_paths, labels_paths, data_aug=True, resize=False)
        query, query_label = images[0], labels[0]
        positive, positive_label = images[1], labels[1]
        negatives, negatives_label = images[2:], labels[2:]
        negatives = torch.stack(negatives, 0)
        negatives_label = torch.stack(negatives_label, 0)

        return query, positive, negatives, query_label, positive_label, negatives_label

    def __len__(self):
        return self.db_struct.num_queries

    def __str__(self):
        return self.info


class DADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.images_paths = sorted(glob(f"{dataset_path}/**/*.jpg", recursive=True))
        self.info = f"DADataset with {len(self.images_paths)} images"

    def __getitem__(self, index):
        image = self.images_paths[index]
        image, label = transform([image], [None], data_aug=False, resize=True, new_dim=(768, 432))
        return image[0]

    def __len__(self):
        return len(self.images_paths)

    def __str__(self):
        return self.info


class SemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_images=500):
        super().__init__()
        self.images_paths = random.sample(sorted(glob(f"{dataset_path}/**/*.jpg", recursive=True)), k=num_images)
        self.info = f"SemDataset {dataset_path} ({len(self.images_paths)})"

    def __getitem__(self, index):
        image = self.images_paths[index]
        label = image.replace('/gallery/', '/gallery_labels/').replace('.jpg', '.png')
        if not exists(label):
            raise Exception(f'Label {label} not found!')
        return sem_transform(image, label)

    def __len__(self):
        return len(self.images_paths)

    def __str__(self):
        return self.info

