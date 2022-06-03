import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import skimage.io


class JacquardDataset(Dataset):
    def __init__(self, dataset_path, n_classes=18, transforms=None, rgd=False):
        self.dataset_path = dataset_path  # path to original Cornell dataset
        self.transforms = transforms  # list of transformations to apply to image and bboxes
        self.n_classes = n_classes  # no. of rotation classes
        self.rgd = rgd

        print("Loading dataset...")
        self.class_list = self.generate_classes()  # class id : [min_theta_val, max_theta_val]
        self.img_list, self.grasp_list = self.generate_data()
        print("Dataset has been loaded.")

    def __getitem__(self, idx):
        img_path = self.img_list[idx]  # the path to the image
        grasps = self.grasp_list[idx]  # the grasps of image in 5D pose format a.k.a (x, y, w, h, theta, theta_class)

        img = Image.open(img_path).convert("RGB")

        # open as RGD image
        if self.rgd:
            # open depth image and convert to [0, 255] format
            depth_path = self.img_list[idx].replace("_RGB.png", "_perfect_depth.tiff")
            depth = Image.fromarray(self.scale_values(skimage.io.imread(depth_path))).convert("L")
            r, g, b = img.split()
            img = Image.merge('RGB', (r, g, depth))  # create RGD image

        # convert grasp to bbox VOC format a.k.a [x_min, y_min, x_max, y_max]
        boxes, labels = [], []
        for i in grasps:
            xmin = i[0] - (i[2] / 2)
            xmax = i[0] + (i[2] / 2)
            ymin = i[1] - (i[3] / 2)
            ymax = i[1] + (i[3] / 2)
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
        rot_start = -np.pi / 2
        rot_size = (rot_range / self.n_classes) * (np.pi / 180)  # how much it rotates between each step (radians)
        for i in range(self.n_classes):
            min_rot, max_rot = rot_start + (i * rot_size), rot_start + ((i + 1) * rot_size)
            class_list[i + 1] = [min_rot, max_rot]
        return class_list

    def generate_data(self):
        img_list = []
        grasp_list = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if f.endswith('_RGB.png'):
                    sample_name = os.path.splitext(f)[0][:-(len('_RGB'))]
                    path = os.path.join(subdir, sample_name)
                    annot_path = path + '_grasps.txt'
                    img_path = path + '_RGB.png'

                    grasps = []
                    with open(annot_path) as file:
                        lines = file.readlines()
                        for i, l in enumerate(lines):
                            ls = l.strip().split(';')
                            x, y, w, h, t = float(ls[0]), float(ls[1]), float(ls[3]), float(ls[4]), float(ls[2])
                            t = t * (np.pi / 180)  # convert to radians
                            c = self.convert_to_class(t)
                            grasps.append([x, y, w, h, t, c])
                    img_list.append(img_path)
                    grasp_list.append(grasps)
        return img_list, grasp_list

    # assign a rotation class to each grasp pose rotation (theta)
    def convert_to_class(self, theta):
        for key, value in self.class_list.items():
            if value[0] <= theta < value[1]:
                return key
        return self.n_classes

    def scale_values(self, img_depth, new_min=0.0, new_max=255.0):
        ''' Function to scale values between 0 and 255 but only calculate min and max from cropped image'''
        img_min, img_max = np.min(img_depth), np.max(img_depth)
        # img_min = 0 - try play with minimum value

        return (((img_depth - img_min) * (new_max - new_min)) / (img_max - img_min)) + new_min


    def visualise_sample(self, idx=None):
        """ Visualise a data-sample without any pre-processing carried out. """
        if idx is None:
            idx = random.randint(0, len(self.img_list) - 1)

        img = Image.open(self.img_list[idx]).convert("RGB")
        grasps = self.grasp_list[idx]

        # open as RGD image
        if self.rgd:
            # open depth image and convert to [0, 255] format
            depth_path = self.img_list[idx].replace("_RGB.png", "_perfect_depth.tiff")
            depth = Image.fromarray(self.scale_values(skimage.io.imread(depth_path))).convert("L")
            r, g, b = img.split()
            img = Image.merge('RGB', (r, g, depth))  # create RGD image

        fig, ax = plt.subplots()
        plt.imshow(img)
        for (x, y, w, h, t, c) in grasps:
            w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
                        h / 2) * np.cos(t)
            bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
            br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
            plt.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
            plt.plot([br_x, tr_x], [br_y, tr_y], c='black')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none',
                                     transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
            ax.add_patch(rect)

        plt.show()


if __name__ == '__main__':
    dataset_path = '../dataset/jacquard'  # cornell dataset folder
    dataset = JacquardDataset(dataset_path, rgd=True)
    dataset.visualise_sample()
