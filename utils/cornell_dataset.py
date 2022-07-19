import math
import os
import random
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from matplotlib.transforms import Affine2D
from torch.utils.data import Dataset
from tqdm import tqdm
import utils.transforms as T
from config import *


class CornellDataset(Dataset):
    def __init__(self, cornell_dataset_path, n_classes=18, transforms=None, depth_path=None, img_format="RGD"):
        """ Initialises a dataset object of the Cornell Grasping dataset.
           :param cornell_dataset_path: (str) path to Cornell Grasping dataset root directory.
           :param n_classes: (int) number of rotation classes
           :param transforms: (list) list of transformations to apply to images and grasps.
           :param depth_path: (str) path to the depth dataset.
           :param img_format: (str) specify whether you want "RGB" or "RGD" images.
        """
        self.dataset_path = cornell_dataset_path
        self.transforms = transforms
        self.n_classes = n_classes
        self.img_format = img_format

        # if no depth path was specified, assume it is in the original Cornell Grasping dataset by default
        self.depth_path = depth_path
        if self.depth_path is None:
            self.depth_path = os.path.join(self.dataset_path, 'depth')

        print("[INFO] Loading Cornell Grasping dataset...")
        self.class_list = self.generate_classes()  # creates a mapping of rotation class idxs to theta values
        self.img_list, self.grasp_list = self.generate_data()  # gets a list of image paths and their GT grasp poses
        print("[INFO] Cornell Grasping dataset has been loaded.")

    def __getitem__(self, idx):
        """ Loads and returns a sample (image and grasps) from the dataset at the given index idx (int). """
        img_path = self.img_list[idx]  # the path to the image
        grasps = self.grasp_list[idx]  # the grasps of image in 5D pose format a.k.a (x, y, w, h, theta, theta_class)
        img = Image.open(img_path).convert("RGB")

        # if RGD format has been selected
        if self.img_format == "RGD":
            depth_path = os.path.join(self.depth_path, os.path.splitext(os.path.basename(img_path))[0].rstrip('r') + 'd.png')
            depth_img = Image.open(depth_path).convert('L')
            r, g, b = img.split()
            img = Image.merge('RGB', (r, g, depth_img))  # create RGD image

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
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:  # apply transformations
            img, target = self.transforms([img, target])

        return img, target

    def generate_classes(self):
        """ Returns a dictionary that maps theta (i.e. rotation) values to a suitable rotation idx. Note that the idx 0
        represents an invalid grasp. Note the minimum theta value can be -pi/2 and the maximum theta value can be pi/2.
           :return class_list: (dict) dictionary where idxs are mapped to unique rotation intervals. Each item in the
           dictionary has the form 'idx':[min_theta_val, max_theta_val]. Hence, a theta value that is >= "min_theta_val"
           and < "max_theta_val" of one of the rotation classes is assigned the "idx" of that particular class.
        """
        class_list = {}
        rot_range = 180  # the range of rotation values
        rot_start = -np.pi / 2
        rot_size = (rot_range / self.n_classes) * (np.pi / 180)  # diff. in rotation between each idx (radians)
        for i in range(self.n_classes):
            min_rot, max_rot = rot_start + (i * rot_size), rot_start + ((i + 1) * rot_size)
            class_list[i + 1] = [min_rot, max_rot]
        return class_list

    def generate_data(self):
        """ Returns a list of image paths in the dataset and lists of labelled grasp poses for each image.
          :return img_list: (list) list of unique images from the dataset.
          :return grasp_list: (list) lists of grasp poses for each image in "img_list". For instance, grasp_list[0]
          is a list of grasp poses for the image in img_list[0]. Note that each grasp poses is stored in the format
          (x, y, w, h, t, c) where (x, y), w, h is the centre point, width and height of the grasp rectangle, and t, c
          is the theta (i.e. rotation) value and rotation class (derived from "class_list").
       """
        img_list, grasp_list = [], []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if f.endswith('r.png'):
                    sample_name = os.path.splitext(f)[0].rstrip('r')  # get unique name of sample
                    path = os.path.join(subdir, sample_name)  # path to sample
                    img_path = path + 'r.png'  # path to image
                    annot_path = path + 'cpos.txt'  # path to image's respective labelled grasps
                    grasps = []  # stores all labelled grasp poses in this image and their rotation class
                    with open(annot_path) as file:
                        lines = file.readlines()
                        grasp_rect = []  # to store the vertices of a single grasp rectangle
                        for i, l in enumerate(lines):
                            xy = l.strip().split()  # parse the (x,y) co-ordinates of each vertex
                            grasp_rect.append((float(xy[0]), float(xy[1])))
                            # once all 4 vertices have been collected
                            if (i + 1) % 4 == 0:
                                if not np.isnan(grasp_rect).any():
                                    # get the parameters (x, y, w, h, t, c) of the grasp rectangle and add to "grasps"
                                    cx, cy, w, h, theta = self.convert_to_5D_pose(grasp_rect)
                                    grasps.append((cx, cy, w, h, theta, self.convert_to_class(theta)))
                                    grasp_rect = []  # reset current grasp rectangle since 4 vertices have been read
                    img_list.append(img_path)
                    grasp_list.append(grasps)
        return img_list, grasp_list

    def convert_to_class(self, theta):
        """ Assigns a given rotation value (in radians) to a suitable rotation class idx.
          :param theta: (float) a rotation value in radians.
          :return: (int) an assigned rotation class.
       """
        for key, value in self.class_list.items():
            if value[0] <= theta < value[1]:
                return key
        return self.n_classes

    def convert_to_5D_pose(self, bbox):
        """ Given four (x,y) vertices of a grasp rectangle, returns the (x, y, w, h, t) parameters of the grasp pose.
        Note that references include https://www.sciencedirect.com/science/article/pii/S0921889021000427 and
        https://github.com/skumra/robotic-grasping/blob/master/utils/dataset_processing/grasp.py.
           :param bbox: (list) a grasp rectangle as a list of four vertices where each vertex is in (x, y) format.
           :return key: (tuple) a tuple (x, y, w, h, t) denoting the 'x, y' centre point, 'w' width, 'h' height and
            't' rotation of the given grasp rectangle. Note that theta is calculated to be in the range [-pi/2, pi/2].
        """
        x1, x2, x3, x4 = bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]
        y1, y2, y3, y4 = bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]
        cx, cy = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
        w = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
        h = np.sqrt(np.power((x3 - x2), 2) + np.power((y3 - y2), 2))
        theta = (np.arctan2((y2 - y1), (x2 - x1)) + np.pi / 2) % np.pi - np.pi / 2  # calculate theta [-pi/2, pi/2]
        return round(cx, 3), round(cy, 3), round(w, 3), round(h, 3), round(theta, 5)

    def get_img_path(self, idx):
        """ Returns the image path (str) of a given sample idx from the dataset. """
        return self.img_list[idx]

    def get_class_mapping(self):
        """ Returns the dictionary (dict) that maps theta rotation values to a rotation class. """
        return self.class_list

    def set_transforms(self, transforms):
        """ Changes the lists of transformations applied to the images and grasp rectangles during training. """
        self.transforms = transforms

    def __len__(self):
        """ Returns the number of samples (int) in the dataset. """
        return len(self.img_list)

    def generate_depth_data(self):
        """ Generates a folder of depth images for each RGB image in the dataset. """
        # create a depth folder if one doesn't exist
        if not os.path.isdir(self.depth_path):
            os.mkdir(self.depth_path)
        print(f"[INFO] Generating Cornell Grasping depth images in the folder: {self.depth_path}")

        def inpaint(img, missing_value=0):
            """ Inpaint missing values in depth image. Note this was taken directly from
            https://github.com/skumra/robotic-grasping/blob/master/utils/dataset_processing/image.py ."""
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            mask = (img == missing_value).astype(np.uint8)
            scale = np.abs(img).max()
            img = img.astype(np.float32) / scale
            img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
            img = img[1:-1, 1:-1]
            img = img * scale
            return img

        def scale_values(img, new_min=0.0, new_max=255.0):
            ''' Scales depth values to [0, 255] but only from a cropped perspective (without background depth included). '''
            cropped = img[101:416, 182:497]
            img_min, img_max = np.min(cropped), np.max(cropped)
            return np.clip((((img - img_min) * (new_max - new_min)) / (img_max - img_min)) + new_min, 0.0, 255.0)

        # creates a depth image for each sample in the dataset in the "new_depth_path" folder
        for img_path in tqdm(self.img_list, desc="Depth files generated"):
            dir_path = os.path.dirname(img_path)
            sample_name = os.path.splitext(os.path.basename(img_path))[0].rstrip('r')
            pcd_path = os.path.join(dir_path, sample_name + '.txt')  # path to point cloud data
            depth_path = os.path.join(self.depth_path, sample_name + 'd.png')

            with open(pcd_path, "r") as pcd_file:
                lines = [line.strip().split(" ") for line in pcd_file.readlines()]

            img_height, img_width = 480, 640
            is_data = False
            img_depth = np.zeros((img_height, img_width), dtype='f8')
            for line in lines:
                if line[0] == 'DATA':  # loop until end of header
                    is_data = True
                    continue
                if is_data:
                    i = int(line[4])
                    col = i % img_width
                    row = math.floor(i / img_width)
                    x, y, z = float(line[0]), float(line[1]), float(line[2])
                    img_depth[row, col] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            img_depth = inpaint(img_depth / 1000)  # in paint missing values
            img_depth = scale_values(img_depth)  # scale image values between 0 and 255
            # save depth image
            img_depth_file = Image.fromarray(img_depth).convert('L')
            img_depth_file.save(depth_path)
        print(f"[INFO] Finished generating depth data.")

    def visualise_sample(self, idx=None, preprocessed=False):
        """ Visualise a data-sample with or without any pre-processing.
           :param idx: (int) a sample idx to view from the dataset. If none is specified a random one is chosen.
           :param preprocessed: (bool) specify whether to view sample before (False) or after (True) preprocessing and
           transforms have been applied.
         """
        def plot_grasp(ax, grasp_pose):
            """ Plots a single given grasp pose (x, y, w, h, t) on a given axes "ax". """
            x, y, w, h, t = grasp_pose
            w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
                    h / 2) * np.cos(t)
            bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
            br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
            plt.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
            plt.plot([br_x, tr_x], [br_y, tr_y], c='black')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none',
                                     transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
            ax.add_patch(rect)

        # choose a random sample idx if idx not specified
        if idx is None:
            idx = random.randint(0, len(self.img_list) - 1)

        fig, ax = plt.subplots()
        print(f"[INFO] Plotting sample {idx} from the Cornell Grasping dataset.")

        # plots an image and it's corresponding grasp poses
        if not preprocessed:
            img = Image.open(self.img_list[idx]).convert("RGB")
            grasps = self.grasp_list[idx]

            for (x, y, w, h, t, c) in grasps:
                plot_grasp(ax, (x, y, w, h, t))
            plt.title("Original sample from the Cornell Grasping dataset")
        else:
            img, targets = self.__getitem__(idx)
            if torch.is_tensor(img):
                img = torchvision.transforms.ToPILImage()(img)  # convert back to normal image

            for b, (xmin, ymin, xmax, ymax) in enumerate(targets['boxes']):
                w, h = xmax - xmin, ymax - ymin
                x, y = xmax - (w / 2), ymax - (h / 2)
                t_range = self.class_list[targets['labels'][b].item()]  # range of theta values [min_t, max_t]
                t = (t_range[0] + t_range[1]) / 2
                plot_grasp(ax, (x, y, w, h, t))
            plt.title("Pre-processed sample from the Cornell Grasping dataset")
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    dataset = CornellDataset(CORNELL_PATH, img_format=IMG_FORMAT)
    dataset.visualise_sample(91)
    dataset.set_transforms(T.get_transforms("cornell", dataset.get_class_mapping()))
    dataset.visualise_sample(91, preprocessed=True)

    # OPTIONAL - generate depth data (uncomment and run if needed)
    # dataset.generate_depth_data()
