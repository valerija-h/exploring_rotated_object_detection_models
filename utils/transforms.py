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
    theta = (class_mapping[theta_class][0] + class_mapping[theta_class][1]) / 2
    return -theta


#############################################################################
# --------------------- BASELINE MODEL TRANSFORMATIONS -------------------- #
#############################################################################


class RandomRotate(object):
    """ Applies a random rotation to the image.
    :param class_mapping (dict): a dictionary mapping classes to theta/rotation values (see Dataset objects).
    :param angle_max (float): the max amount of degrees to rotate (in degrees)
    """
    def __init__(self, class_mapping, angle_max=360):
        self.angle_max = angle_max
        self.class_mapping = class_mapping

    def __call__(self, data):
        img, target = data[0], data[1]  # obtain the image and grasp poses
        angle = random.randint(0, self.angle_max)  # select a random num of degrees to rotate

        img = self.rotate_image(img, angle)  # rotate the image
        target = self.rotate_bboxes(target, angle, img)  # rotate the grasp rects based on image

        return img, target

    def rotate_image(self, img, angle):
        """ Rotates a given image about its centre by a specified num of degrees (i.e. angle) clockwise. """
        img = img.rotate(angle=angle)
        return img

    def rotate_bboxes(self, target, angle, img):
        """ Rotates grasp poses about the img centre by a specified num of degrees (i.e. angle) clockwise. """
        bboxes, theta_classes = target['boxes'], target['labels']  # obtain grasp rects and their rotation class
        # get image properties
        w, h = img.size
        cx, cy = w // 2, h // 2
        # get corners of the grasp rects and apply a rotation matrix to get rotated (i.e. transformed) corner points
        corners = self.get_corners(bboxes, theta_classes)
        rotated_bboxes = self.rotate_box(corners, angle, cx, cy)
        # recalculate new theta and co-ords in VOC format and assign rotation class
        new_bboxes, thetas = self.re_calculate_bbox(rotated_bboxes)
        target['boxes'] = np.asarray(new_bboxes, dtype='float32')
        target['labels'] = np.asarray([re_assign_class(t, self.class_mapping) for t in thetas], dtype='int64')
        return target

    def get_corners(self, bboxes, theta_classes):
        """ Get the corners of current grasp rectangles.
        :param bboxes: (list) the grasp rectangles of shape N x 4 where N is the num of grasp rectangles represented in
        the format [x1, y1, x2, y2].
        :param theta_classes: (list) the rotation class of each grasp rectangles of shape N.
        :return corners: (np.ndarray) the corners of the grasp rectangles in shape N x 8  where N is the num of grasp
        rectangles represented in the format [x1, y1, x2, y2, x3, y3, x4, y4].
        """
        w = (bboxes[:, 2] - bboxes[:, 0])  # width of bbox
        h = (bboxes[:, 3] - bboxes[:, 1])  # height of bbox
        x, y = bboxes[:, 0] + (w // 2), bboxes[:, 1] + (h // 2)  # center of bbox
        t = [((self.class_mapping[t][0] + self.class_mapping[t][1]) / 2) for t in theta_classes]  # calculate theta
        w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (h / 2) * np.cos(t)
        bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
        br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
        x1, y1 = tl_x.reshape(-1, 1), tl_y.reshape(-1, 1)
        x2, y2 = tr_x.reshape(-1, 1), tr_y.reshape(-1, 1)
        x3, y3 = br_x.reshape(-1, 1), br_y.reshape(-1, 1)
        x4, y4 = bl_x.reshape(-1, 1), bl_y.reshape(-1, 1)
        # get the corners of each grasp rectangle into shape N x 8
        corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
        return corners

    def rotate_box(self, corners, angle, cx, cy):
        """ Rotate the grasping rectangles about the center of the image.
        :param corners: (np.ndarray) an array of N grasp rectangles of shape N x 8 where each grasp rectangle is defined
        by their corner co-ordinates [x1, y1, x2, y2, y3, x4, y4].
        :param angle: (float) angle by which the image was rotated.
        :param cx: (int) x co-ordinate centre of image.
        :param cy: (int) y co-ordinate centre of image.
        :return calculated: (np.ndarray) the corners of the grasp rectangles after being rotated of shape N x 8 where
        each grasp rectangle is defined by their corner co-ordinates [x1, y1, x2, y2, y3, x4, y4].
        """
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        calculated = np.dot(M, corners.T).T
        calculated = calculated.reshape(-1, 8)
        return calculated

    def re_calculate_bbox(self, bboxes):
        """ Recalculate the co-ordinates of the grasp rectangle in VOC format [xmin, ymin, xmax, ymax] and calculate
         the new theta value (in radians).
          :param bboxes: (np.ndarray) the corners of the rotated grasp rectangles of shape N x 8 where N is the num of
          grasp rectangles represented by the format [x1, y1, x2, y2, y3, x4, y4].
          :return new_bbox: (np.ndarray) the new co-ordinates of the grasp rectangles in Pascal VOC format in the shape
          N x 4 where N is num of grasp rectangles in the format [xmin, ymin, xmax, ymax].
          :return theta: (np.ndarray) the new thetas (i.e. rotation values) of the rotated grasp rectangles.
        """
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


class RandomShift(object):
    """ Applies a random horizontal or vertical shift to the image.
      :param px: (float) the max amount of pixels the image can shift.
      :param shift: (str) specify whether to shift horizontally (x), vertically (y) or both.
    """
    def __init__(self, px=50, shift='both'):
        self.px = px
        self.shift = shift

    def __call__(self, data):
        img, target = data[0], data[1]
        y_shift, x_shift = 0, 0
        # choose a random amount of pixels to translate in chosen directions
        if self.shift == 'x' or self.shift == 'both':
            x_shift = random.randint(-self.px, self.px + 1)
        if self.shift == 'y' or self.shift == 'both':
            y_shift = random.randint(-self.px, self.px + 1)
        # translate image by num. of pixels
        translated_image = img.transform(img.size, AFFINE, (1, 0, x_shift, 0, 1, y_shift))
        # translate bboxes by num. of pixels
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
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (
                    target['boxes'][:, 2] - target['boxes'][:, 0])
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
        # target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        # crop the center of the image
        img = img.crop((self.left, self.top, right, bottom))

        return img, target


class ToTensor(object):
    """ Convert image and bounding boxes to Tensors. """

    def __call__(self, data):
        img, target = data[0], data[1]
        new_img = torchvision.transforms.ToTensor()(img).float()
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        return new_img, target


def visualise_transforms(data_loader, class_mapping):
    images, targets = next(iter(data_loader))

    for i in range(len(images)):
        # convert the image to
        fig, ax = plt.subplots()
        image = torchvision.transforms.ToPILImage()(images[i])
        ax.imshow(image)
        for b, (xmin, ymin, xmax, ymax) in enumerate(targets[i]['boxes']):
            w, h = xmax - xmin, ymax - ymin
            x, y = xmax - (w / 2), ymax - (h / 2)
            t_range = class_mapping[targets[i]['labels'][b].item()]  # range of theta values [min_t, max_t]
            t = (t_range[0] + t_range[1]) / 2
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
