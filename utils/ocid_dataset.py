import os
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from matplotlib.transforms import Affine2D
from torch.utils.data import Dataset
import utils.transforms as T
from config import *

class OCIDDataset(Dataset):
    def __init__(self, ocid_dataset_path, n_classes=18, transforms=None, img_format="RGD"):
        """ Initialises a dataset object of the OCID Grasping dataset.
           :param ocid_dataset_path: (str) path to OCID Grasping dataset root directory.
           :param n_classes: (int) number of rotation classes
           :param transforms: (list) list of transformations to apply to images and grasps.
           :param img_format: (str) specify whether you want "RGB" or "RGD" images.
        """
        self.dataset_path = ocid_dataset_path
        self.transforms = transforms
        self.n_classes = n_classes
        self.img_format = img_format

        print("[INFO] Loading OCID Grasping dataset...")
        self.class_list = self.generate_classes()  # creates a mapping of rotation class idxs to theta values
        self.img_list, self.grasp_list = self.generate_data()  # gets a list of image paths and their GT grasp poses
        print("[INFO] OCID Grasping dataset has been loaded.")

    def __getitem__(self, idx):
        """ Loads and returns a sample (image and grasps) from the dataset at the given index idx (int). """
        img_path = self.img_list[idx]  # the path to the image
        grasps = self.grasp_list[idx]  # the grasps of image in 5D pose format a.k.a (x, y, w, h, theta, theta_class)
        img = Image.open(img_path).convert("RGB")

        # open as RGD image
        if self.img_format == "RGD":
            # open depth image and convert to [0, 255] format
            depth_img = Image.open(self.img_list[idx].replace("rgb", "depth"))
            depth_img = Image.fromarray(self.scale_values(np.asarray(depth_img))).convert('L')
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
            "images_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms([img, target])

        return img, target

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

    def scale_values(self, img_depth, new_min=0.0, new_max=255.0):
        ''' Function to scale values between 0 and 255 but only calculate min and max from cropped image '''
        img_min, img_max = np.min(img_depth), np.max(img_depth)
        return (((img_depth - img_min) * (new_max - new_min)) / (img_max - img_min)) + new_min

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
        img_list = []
        grasp_list = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for f in files:
                if "rgb" in subdir and f.endswith('.png'):
                    sample_name = os.path.splitext(f)[0]
                    path = os.path.join(subdir, sample_name)
                    annot_path = path.replace("rgb", "Annotations") + '.txt'
                    img_path = path + '.png'

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
        h = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
        w = np.sqrt(np.power((x3 - x2), 2) + np.power((y3 - y2), 2))
        theta = (np.arctan2((y2 - y1), (x2 - x1))) % np.pi - np.pi / 2  # calculate theta [-pi/2, pi/2]
        return round(cx, 3), round(cy, 3), round(w, 3), round(h, 3), round(theta, 5)

    def convert_to_class(self, theta):
        """ Assigns a given rotation value (in radians) to a suitable rotation class idx.
          :param theta: (float) a rotation value in radians.
          :return: (int) an assigned rotation class.
       """
        for key, value in self.class_list.items():
            if value[0] <= theta < value[1]:
                return key
        return self.n_classes


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
        print(f"[INFO] Plotting sample {idx} from the OCID Grasping dataset.")

        # plots an image and it's corresponding grasp poses
        if not preprocessed:
            img = Image.open(self.img_list[idx]).convert("RGB")
            grasps = self.grasp_list[idx]

            for (x, y, w, h, t, c) in grasps:
                plot_grasp(ax, (x, y, w, h, t))
            plt.title("Original sample from the OCID Grasping dataset")
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
            plt.title("Pre-processed sample from the OCID Grasping dataset")
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    dataset = OCIDDataset(OCID_PATH, img_format=IMG_FORMAT)
    dataset.visualise_sample(1143)
    dataset.set_transforms(T.get_transforms("ocid", dataset.get_class_mapping()))
    dataset.visualise_sample(1143, preprocessed=True)

