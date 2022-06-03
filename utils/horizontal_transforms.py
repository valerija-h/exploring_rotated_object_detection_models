import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from PIL.Image import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, AFFINE
import random
import cv2


def re_assign_class(theta, class_mapping):
    for key, value in class_mapping.items():
        if value[0] <= theta < value[1]:
            return key
    return class_mapping[len(class_mapping)]

def invert_angle(theta_class, class_mapping):
    theta = (class_mapping[theta_class][0] + class_mapping[theta_class][1])/2
    return -theta


class RandomRotate(object):
    """ Randomly translates an object
    Args:
    angle_max (float): The max amount of degrees to rotate (in degrees)
    """
    def __init__(self, class_mapping, angle_max=360):
        self.angle_max = angle_max
        self.class_mapping = class_mapping

    def __call__(self, data):
        img, target = data[0], data[1]
        angle = random.randint(0, 360)

        img = self.rotate_image(img, angle)
        w, h = img.size
        cx, cy = w//2, h//2

        target = self.rotate_bboxes(target, angle, cx, cy, w, h)

        return img, target

    def rotate_image(self, img, angle):
        img = img.rotate(angle=angle)
        return img

    def rotate_bboxes(self, target, angle, cx, cy, w, h):
        bboxes, theta_classes = target['boxes'], target['labels']
        # get corners of bboxes and apply rotation matrix to get rotated corners
        corners = self.get_corners(bboxes, theta_classes)
        rotated_bboxes = self.rotate_box(corners, angle, cx, cy, w, h)
        # recalculate theta, get new VOC format co-ords and re-assign class
        new_bboxes, thetas = self.re_calculate_bbox(rotated_bboxes)
        target['boxes'] = np.asarray(new_bboxes, dtype='float32')
        target['labels'] = np.asarray([re_assign_class(t, self.class_mapping) for t in thetas], dtype='int64')
        return target

    # link to calculate cx, cy, w, h, theta after - https://www.sciencedirect.com/science/article/pii/S0921889021000427
    def re_calculate_bbox(self, bboxes):
        # bl then br then tr then tl - have to put in anti-clockwise
        x1, x2, x3, x4 = bboxes[:, 6], bboxes[:, 4], bboxes[:, 2], bboxes[:, 0]
        y1, y2, y3, y4 = bboxes[:, 7], bboxes[:, 5], bboxes[:, 3], bboxes[:, 1]
        theta = (np.arctan2((y2 - y1), (x2 - x1)) + np.pi / 2) % np.pi - np.pi / 2  # calculate theta [-pi/2, pi/2]
        cx, cy = (x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4
        w = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
        h = np.sqrt(np.power((x3 - x2), 2) + np.power((y3 - y2), 2))
        xmin, xmax = (cx - (w / 2)).reshape(-1, 1), (cx + (w / 2)).reshape(-1, 1)
        ymin, ymax = (cy - (h / 2)).reshape(-1, 1), (cy + (h / 2)).reshape(-1, 1)
        new_bbox = np.hstack((xmin, ymin, xmax, ymax))
        return new_bbox, theta



    def get_corners(self, bboxes, theta_classes):
        """ Get corners of current bounding boxes
        Args:
            bboxes: numpy.ndarray
                Numpy array containing bounding boxes of shape `N X 4` where N is the
                number of bounding boxes and the bounding boxes are represented in the
                format `x1 y1 x2 y2`
        """
        w = (bboxes[:, 2] - bboxes[:, 0])
        h = (bboxes[:, 3] - bboxes[:, 1])
        x, y = bboxes[:, 0] + (w//2), bboxes[:, 1] + (h//2)  # center of bbox
        t = [((self.class_mapping[t][0] + self.class_mapping[t][1])/2) for t in theta_classes]
        w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (h / 2) * np.cos(t)
        bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
        br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
        x1, y1 = tl_x.reshape(-1, 1), tl_y.reshape(-1, 1)
        x2, y2 = tr_x.reshape(-1, 1), tr_y.reshape(-1, 1)
        x3, y3 = br_x.reshape(-1, 1), br_y.reshape(-1, 1)
        x4, y4 = bl_x.reshape(-1, 1), bl_y.reshape(-1, 1)

        corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

        return corners

    def rotate_box(self, corners, angle, cx, cy, w, h):
        """Rotate the bounding box.
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        angle : float
            angle by which the image is to be rotated
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
        h : int
            height of the image
        w : int
            width of the image
        """

        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # cos = np.abs(M[0, 0])
        # sin = np.abs(M[0, 1])

        # nW = int((h * sin) + (w * cos))
        # nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        # M[0, 2] += (nW / 2) - cx
        # M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M, corners.T).T

        calculated = calculated.reshape(-1, 8)

        return calculated



class RandomShift(object):
    """ Randomly translates an object
    Args:
    px (float): The amount the image can shift in x, y direction
    shift (str): 'x', 'y', 'both'
    """
    def __init__(self, px=50, shift='both'):
        self.px = px
        self.shift = shift

    def __call__(self, data):
        img, target = data[0], data[1]
        y_shift, x_shift = 0, 0

        if self.shift == 'x' or self.shift == 'both':
            x_shift = random.randint(-self.px, self.px+1)
        if self.shift == 'y' or self.shift == 'both':
            y_shift = random.randint(-self.px, self.px+1)

        # translate image
        translated_image = img.transform(img.size, AFFINE, (1, 0, x_shift, 0, 1, y_shift))
        # translate bbox
        target['boxes'][:, 0] -= x_shift
        target['boxes'][:, 2] -= x_shift
        target['boxes'][:, 1] -= y_shift
        target['boxes'][:, 3] -= y_shift
        return translated_image, target

class RandomHorizontalFlip(object):
    """ Randomly horizontally flips the Image with the probability *p*
    Args:
    p (float): The probability with which the image is flipped
    """
    def __init__(self, class_mapping, p=0.5):
        self.class_mapping = class_mapping
        self.p = p

    def __call__(self, data):
        img, target = data[0], data[1]
        img_center = np.array((img.size[1], img.size[0]))[::-1] / 2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img.transpose(FLIP_LEFT_RIGHT)
            target['boxes'][:, [0, 2]] += 2 * (img_center[[0, 2]] - target['boxes'][:, [0, 2]])
            box_w = abs(target['boxes'][:, 0] - target['boxes'][:, 2])

            target['boxes'][:, 0] -= box_w
            target['boxes'][:, 2] += box_w

            for i, l in enumerate(target['labels']):
                target['labels'][i] = re_assign_class(invert_angle(l, self.class_mapping), self.class_mapping)
        return img, target

class RandomVerticalFlip(object):
    """ Randomly vertically flips the Image with the probability *p*
    Args:
    p (float): The probability with which the image is flipped
    """
    def __init__(self, class_mapping, p=0.5):
        self.class_mapping = class_mapping
        self.p = p

    def __call__(self, data):
        img, target = data[0], data[1]
        img_center = np.array((img.size[1], img.size[0]))[::-1] / 2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img.transpose(FLIP_TOP_BOTTOM)
            target['boxes'][:, [1, 3]] += 2 * (img_center[[1, 3]] - target['boxes'][:, [1, 3]])

            box_h = abs(target['boxes'][:, 1] - target['boxes'][:, 3])

            target['boxes'][:, 1] -= box_h
            target['boxes'][:, 3] += box_h
            for i, l in enumerate(target['labels']):
                target['labels'][i] = re_assign_class(invert_angle(l, self.class_mapping), self.class_mapping)
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

        # recalculate new area - I don't think you need to recalc it?
        #target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
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
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        return new_img, target

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