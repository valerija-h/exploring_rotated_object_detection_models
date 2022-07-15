''' Convert the Dataset into DOTA format'''

import os
import torch
import torchvision
import time
from torch.utils.data import DataLoader
from utils.cornell_dataset import CornellDataset
from utils.ocid_dataset import OCIDDataset
from utils.jacquard_dataset import JacquardDataset
from shapely.geometry import Polygon

from utils import transforms as T
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np

# set seeds for reproducibility
torch.manual_seed(0)
#
# datasets = [
#     CornellDataset('../dataset/cornell/rgd'),
#     JacquardDataset('../dataset/jacquard'),
#     OCIDDataset('../dataset/ocid')
# ]

# data preprocessing parameters
TEST_SPLIT = 0.20  # percentage of test samples from all samples
VAL_SPLIT = 0.10  # percentage of validation samples from training samples
SEED_SPLIT = 42

# split the dataset into training, testing and validation sets
def split_dataset(dataset):
    test_size = round(TEST_SPLIT * len(dataset))
    train_size = len(dataset) - test_size
    val_size = round(VAL_SPLIT * train_size)
    train_size = train_size - val_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size],
                                                                             generator=torch.Generator().manual_seed(SEED_SPLIT))
    return train_dataset, test_dataset, val_dataset

def VOC_to_DOTA(bbox, t):
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    x, y = xmax - (w / 2), ymax - (h / 2)
    w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
            h / 2) * np.cos(t)
    bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
    br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
    return tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y


if __name__ == '__main__':
    target_path = 'D:\Datasets\Cornell_DOTA'
    dataset = CornellDataset('../dataset/cornell/rgd')
    train_dataset, test_dataset, val_dataset = split_dataset(dataset)
    subdir = ['/train/', '/test/', 'val']
    class_mapping = dataset.get_class_mapping()

    for s, split in enumerate([train_dataset, test_dataset, val_dataset]):
        idxs = split.indices
        for i, idx in enumerate(idxs):
            img, target = dataset.__getitem__(idx)
            img_name = os.path.basename(dataset.get_img_path(idx))
            new_img_path = target_path + subdir[s] + img_name
            new_annot_path = target_path + '/all_labels/' + img_name.replace("r.png", ".txt")

            #for each bbox, write a text file in DOTA format
            annot_lines = []
            for b in range(len(target['labels'])):
                bbox = target['boxes'][b]
                theta = (class_mapping[target['labels'][b]][0] + class_mapping[target['labels'][b]][1]) / 2
                new_line = " ".join(str(int(item)) for item in VOC_to_DOTA(bbox, theta)) + " grasp 0"
                annot_lines.append(new_line)

            # save image and annotation
            new_annot_file = open(new_annot_path, "w")
            for element in annot_lines:
                new_annot_file.write(element + "\n")
            new_annot_file.close()
            img.save(new_img_path)

            print(f'[INFO]: {subdir[s]} - {i}/{len(idxs)} copied - {img_name}')


