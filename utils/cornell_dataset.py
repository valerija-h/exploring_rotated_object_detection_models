import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


class CornellDataset(Dataset):
    def __init__(self, dataset_path, n_classes=18, transforms=None):
        self.dataset_path = dataset_path  # path to original Cornell dataset
        self.transforms = transforms  # list of transformations to apply to image and bboxes
        self.n_classes = n_classes  # no. of rotation classes

        print("Loading dataset...")
        self.class_list = self.generate_classes()  # class id : [min_theta_val, max_theta_val]
        self.img_list, self.grasp_list = self.generate_data()
        print("Dataset has been loaded.")

    def __getitem__(self, idx):
        img_path = self.img_list[idx]  # the path to the image
        grasps = self.grasp_list[idx]  # the grasps of image in 5D pose format a.k.a (x, y, w, h, theta, theta_class)
        img = Image.open(img_path).convert("RGB")

        # convert grasp to bbox VOC format a.k.a [x_min, y_min, x_max, y_max]
        boxes, labels = [], []
        for i in grasps:
            xmin = i[0] - (i[2]/2)
            xmax = i[0] + (i[2]/2)
            ymin = i[1] - (i[3]/2)
            ymax = i[1] + (i[3]/2)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(i[5])

        # keep grasps as np.arrays for transformations
        boxes = np.asarray(boxes, dtype='float32')
        labels = np.asarray(labels, dtype='int64')

        # convert the rest to tensors
        image_id = torch.tensor([idx])
        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.float32)
        iscrowd = torch.zeros((len(grasps),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "images_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms([img, target])

        return img, target

    def __len__(self):
        return len(self.img_list)

    # create a mapping between class idxs and rotation values, note class '0' = invalid proposal
    def generate_classes(self):
        class_list = {}
        rot_range = 180  # the range of rotation, ours is between -pi/2 and pi/2 (degrees)
        rot_start = -np.pi/2
        rot_size = (rot_range/self.n_classes) * (np.pi/180)  # how much it rotates between each step (radians)
        for i in range(self.n_classes):
            min_rot, max_rot = rot_start + (i*rot_size), rot_start + ((i+1)*rot_size)
            class_list[i+1] = [min_rot, max_rot]
        return class_list

    def generate_data(self):
        img_list = []
        grasp_list = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if f.endswith('.png'):
                    path = os.path.join(subdir, os.path.splitext(f)[0].rstrip('r'))
                    img_path = path + 'r.png'  # path to image
                    annot_path = path + 'cpos.txt'  # path to image's respective positive annotations
                    grasps = []  # stores all grasp poses and their class in current image
                    with open(annot_path) as file:
                        lines = file.readlines()
                        grasp_rect = []  # to store the vertices of a single grasp rectangle
                        for i, l in enumerate(lines):
                            # parse the (x,y) co-ordinates of each grasp box vertice
                            xy = l.strip().split()
                            grasp_rect.append((float(xy[0]), float(xy[1])))
                            if (i + 1) % 4 == 0:
                                if not np.isnan(grasp_rect).any():
                                    cx, cy, w, h, theta = self.convert_to_5D_pose(grasp_rect)
                                    grasps.append((cx, cy, w, h, theta, self.convert_to_class(theta)))
                                    grasp_rect = []  # reset current grasp rectangle after 4 vertices have been read
                    img_list.append(img_path)
                    grasp_list.append(grasps)
        return img_list, grasp_list

    # assign a rotation class to each grasp pose rotation (theta)
    def convert_to_class(self, theta):
        for key, value in self.class_list.items():
            if value[0] <= theta < value[1]:
                return key
        return self.n_classes

    # link to calculate cx, cy, w, h, theta after - https://www.sciencedirect.com/science/article/pii/S0921889021000427
    def convert_to_5D_pose(self, bbox):
        x1, x2, x3, x4 = bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]
        y1, y2, y3, y4 = bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]
        cx, cy = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
        w = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
        h = np.sqrt(np.power((x3 - x2), 2) + np.power((y3 - y2), 2))
        theta = (np.arctan2((y2 - y1), (x2 - x1)) + np.pi / 2) % np.pi - np.pi / 2  # calculate theta [-pi/2, pi/2]
        return round(cx, 3), round(cy, 3), round(w, 3), round(h, 3), round(theta, 5)

    def get_class_mapping(self):
        return self.class_list

    def set_transforms(self, transforms):
        self.transforms = transforms

    def visualise_sample(self, idx=None):
        """ Visualise a data-sample without any pre-processing carried out. """
        if idx is None:
            idx = random.randint(0, len(self.img_list)-1)
        img = Image.open(self.img_list[idx]).convert("RGB")
        grasps = self.grasp_list[idx]

        fig, ax = plt.subplots()
        ax.imshow(img)
        for (x, y, w, h, t, c) in grasps:
            w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (h / 2) * np.cos(t)
            bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
            br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
            plt.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
            plt.plot([br_x, tr_x], [br_y, tr_y], c='black')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none',
                                     transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
            ax.add_patch(rect)

        plt.imshow(img)
        plt.show()
    

if __name__ == '__main__':
    dataset_path = '../dataset/cornell/RGB'  # cornell dataset folder
    dataset = CornellDataset(dataset_path)
    dataset.visualise_sample()
