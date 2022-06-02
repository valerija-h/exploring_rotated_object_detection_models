import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from PIL.Image import FLIP_LEFT_RIGHT
import random

class RandomHorizontalFlip(object):
    """ Randomly horizontally flips the Image with the probability *p*
    Args:
    p (float): The probability with which the image is flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        img, target = data[0], data[1]
        img_arr = np.asarray(img)
        img_center = np.array(img_arr.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img.transpose(FLIP_LEFT_RIGHT)
            target['boxes'][:, [0, 2]] += 2 * (img_center[[0, 2]] - target['boxes'][:, [0, 2]])

            box_w = abs(target['boxes'][:, 0] - target['boxes'][:, 2])

            target['boxes'][:, 0] -= box_w
            target['boxes'][:, 2] += box_w

        return img, target

class CentreCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        img, target = data[0], data[1]
        width, height = img.size  # get dimensions
        left = (width - self.output_size[0]) / 2
        top = (height - self.output_size[1]) / 2
        right = (width + self.output_size[0]) / 2
        bottom = (height + self.output_size[1]) / 2

        # scale_width = self.output_size[0]/width
        # scale_height = self.output_size[1]/height
        # target['boxes'][:, 0] *= scale_width  # change xmin
        # target['boxes'][:, 2] *= scale_width  # change xmax
        # target['boxes'][:, 1] *= scale_height  # change ymin
        # target['boxes'][:, 3] *= scale_height  # change yax
        target['boxes'][:, 0] -= left  # change xmin
        target['boxes'][:, 2] -= left  # change xmax
        target['boxes'][:, 1] -= top  # change ymin
        target['boxes'][:, 3] -= top  # change yax

        # recalculate new area
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        # crop the center of the image
        img = img.crop((left, top, right, bottom))

        return img, target


class CustomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, top, left, output_size):
        self.top = top
        self.left = left
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        img, target = data[0], data[1]
        right = self.left + self.output_size[0]
        bottom = self.top + self.output_size[1]

        target['boxes'][:, 0] -= self.left  # change xmin
        target['boxes'][:, 2] -= self.left  # change xmax
        target['boxes'][:, 1] -= self.top  # change ymin
        target['boxes'][:, 3] -= self.top  # change yax

        # recalculate new area
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        # crop the center of the image
        img = img.crop((self.left, self.top, right, bottom))

        return img, target


class Normalize(object):
    """ Convert image and bounding boxes to Tensors. """

    def __call__(self, data):
        img, target = data[0], data[1]
        img /= 255.0
        return img, target

class ZeroCentre(object):
    """ Convert image and bounding boxes to Tensors. """

    def __call__(self, data):
        img, target = data[0], data[1]
        centered = img - img.mean(axis=(0,1,2), keepdims=True)
        return centered, target

class ToTensor(object):
    """ Convert image and bounding boxes to Tensors. """

    def __call__(self, data):
        img, target = data[0], data[1]
        new_img = torchvision.transforms.PILToTensor()(img).float()
        new_target = target
        new_target["boxes"] = torch.as_tensor(new_target["boxes"], dtype=torch.float32)
        return new_img, new_target

def visualise_transforms(data_loader, class_mapping):
    
    images, targets = next(iter(data_loader))

    for i in range(len(images)):
        # convert the image to
        fig, ax = plt.subplots()
        image = image = torchvision.transforms.ToPILImage()(images[i])
        ax.imshow(image)
        for b, (xmin, ymin, xmax, ymax) in enumerate(targets[i]['boxes']):
            w, h = xmax-xmin, ymax-ymin
            x, y = xmax-(w/2), ymax-(h/2)
            t_range = class_mapping[targets[i]['labels'][b].item()] # range of theta values [min_t, max_t]
            t = (t_range[0] + t_range[1])/2
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


def collate_fn(batch):
    """ Needed for data loader to load images with batch size > 1 """
    return tuple(zip(*batch))