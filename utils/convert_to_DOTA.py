
import os
import torch
import time
from utils.cornell_dataset import CornellDataset
from utils.ocid_dataset import OCIDDataset
from config import *
from utils import transforms as T
from tqdm.auto import tqdm
import numpy as np

# set seeds to ensure reproducibility
torch.manual_seed(0)

''' 
This file generates copies of the original grasping datasets in DOTA format for the
rotated object detector networks. 
'''

def VOC_to_DOTA(bbox, t):
    """ Changes a given grasp box in VOC format [xmin, ymin, xmax, ymax] to DOTA format
    [x1, y1, x2, y2, x3, y3, x4, y4].
      :param bbox: (list) a grasp rectangle in VOC format [xmin, ymin, xmax, ymax] (i.e without rotation).
      :param t: (float) the rotation value of bbox (in radians).
      :return: (tuple) a grasp rectangle in DOTA format [x1, y1, x2, y2, x3, y3, x4, y4].
     """
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    x, y = xmax - (w / 2), ymax - (h / 2)
    w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
            h / 2) * np.cos(t)
    bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
    br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
    return tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y


def generate_DOTA(dataset_name, dataset_path, DOTA_path, img_format="RGD"):
    """ Generates copies of images and annotations of a grasping dataset in DOTA format. 
      :param dataset_name: (str) the name of the dataset to copy images and grasps from. Either "cornell" or "ocid".
      :param dataset_path: (str) the root directory of the original dataset (i.e. dataset_name).
      :param DOTA_path: (str) the directory where all DOTA datasets will be placed into.
      :param img_format: (str) the format to save images in. Either "RGD" or "RGB".
    """

    # load the correct Dataset object
    if dataset_name == "cornell":
        dataset = CornellDataset(dataset_path, img_format=img_format) 
    elif dataset_name == "ocid":
        dataset = OCIDDataset(dataset_path, img_format=img_format)

    train_dataset, test_dataset, val_dataset = T.split_dataset(dataset) # split dataset into train, test, val sets
    subdir_name = ['train', 'test', 'val'] # folder names for images of each dataset split
    annot_name = 'all_labels' # folder name containing all annotation files
    class_mapping = dataset.get_class_mapping()
    print(f'[INFO]: Generating DOTA format files for the {dataset_name} Grasping dataset in the folder {os.path.join(DOTA_path, dataset_name)}.')

    # if the subdir annot directory doesn't exist... make it
    if not os.path.exists(os.path.join(DOTA_path, dataset_name, annot_name)):
        print(f'[INFO]: Creating new directory to store DOTA annotations - {os.path.join(DOTA_path, dataset_name, annot_name)}')
        os.makedirs(os.path.join(DOTA_path, dataset_name, annot_name))

    for s, split in enumerate([train_dataset, test_dataset, val_dataset]):
        # if the subdir img directories don't exist... make them
        if not os.path.exists(os.path.join(DOTA_path, dataset_name, subdir_name[s])):
            print(f'[INFO]: Creating new directory to store {subdir_name[s]} images - {os.path.join(DOTA_path, dataset_name, subdir_name[s])}')
            os.makedirs(os.path.join(DOTA_path, dataset_name, subdir_name[s]))
        
        idxs = split.indices # get the sample idxs of each dataset split
        for idx in tqdm(idxs, desc=f"'{subdir_name[s]}' files copied"):
            img, target = dataset.__getitem__(idx)
            img_name = os.path.basename(dataset.get_img_path(idx))

            # OPTIONAL - remove trailing characters for some datasets to have same ID names for samples and i
            if dataset_name == "cornell":
                img_name = img_name.replace("r.png", ".png")

            new_img_path = os.path.join(DOTA_path, dataset_name, subdir_name[s], img_name) # path to store new img
            new_annot_path = os.path.join(DOTA_path, dataset_name, annot_name, img_name.replace(".png", ".txt")) # path to store new label
            
            # for each grasp rectangle, create a line in the annotation file (in DOTA format)
            annot_lines = []
            for b in range(len(target['labels'])):
                bbox = target['boxes'][b]
                theta = (class_mapping[target['labels'][b]][0] + class_mapping[target['labels'][b]][1]) / 2
                new_line = " ".join(str(int(item)) for item in VOC_to_DOTA(bbox, theta)) + " grasp 0"
                annot_lines.append(new_line)

            # save new image and annotation (in DOTA format)
            new_annot_file = open(new_annot_path, "w")
            for element in annot_lines:
                new_annot_file.write(element + "\n")
            new_annot_file.close()
            img.save(new_img_path)
    print(f'[INFO]: Finished generating DOTA format files for the {dataset_name} Grasping dataset.')


if __name__ == '__main__':
    # IMPORTANT - change before running
    DOTA_path = '/home/user/Documents/DOTA_Datasets' # path to folder containing DOTA datasets
    dataset_choice = 'cornell'
    dataset_path = '/home/user/Documents/exploring_rotated_object_detectors_not_official/dataset/cornell/RGD'

    generate_DOTA(dataset_choice, dataset_path, DOTA_path)
