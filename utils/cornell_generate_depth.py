import os
import numpy as np
from PIL import Image
import cv2
import math
import shutil

def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale
    return img

def scale_values(img_depth, new_min=0.0, new_max=255.0):
    ''' Function to scale values between 0 and 255 but only calculate min and max from cropped image'''
    cropped = img_depth[101:416, 182:497]
    img_min, img_max = np.min(cropped), np.max(cropped)
    # img_min = 0 - try play with minimum value

    return np.clip((((img_depth-img_min)*(new_max - new_min))/(img_max-img_min)) + new_min, 0.0, 255.0)

def generate_depth_data(dataset_path):
    for subdir, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.png'):
                sample_id = os.path.splitext(f)[0].rstrip('r')
                rgb_path = os.path.join(subdir, os.path.splitext(f)[0].rstrip('r'))
                depth_path = rgb_path.replace("/RGB/", "/depth/")
                img_path = rgb_path + 'r.png'  # path to image
                depth_path = depth_path + 'd.png'
                pcd_path = rgb_path + '.txt'  # path 
                rgd_path = rgb_path.replace("/RGB/", "/RGD/") + 'r.png'  # path to new RG-D image

                # get PCD file and convert to depth image
                with open(pcd_path, "r") as pcd_file:
                    lines = [line.strip().split(" ") for line in pcd_file.readlines()]
                
                img_height = 480
                img_width = 640
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
                        x, y, z = float(line[0]), float(line[1]),float(line[2])
                        img_depth[row, col] = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                img_depth = inpaint(img_depth/1000) # in paint missing values
                # scale image values between 0 and 255
                img_depth = scale_values(img_depth)
                # save depth and rgd image
                img_depth_file = Image.fromarray(img_depth).convert('L')
                img_depth_file.save(os.path.join(depth_path))
                RGB_image = Image.open(img_path).convert('RGB')
                r, g, b = RGB_image.split()
                RGD_image = Image.merge('RGB', (r, g, img_depth_file)) # create RGD image
                RGD_image.save(rgd_path)
                # need co-ordinates too
                # shutil.copyfile(rgb_path + 'cpos.txt', rgb_path.replace("/RGB/", "/RGD/") + 'cpos.txt')


if __name__ == '__main__':
    dataset_path = '../dataset/cornell/RGB'  # cornell dataset folder
    #dataset = CornellDataset(dataset_path)
    #dataset.visualise_sample()
    generate_depth_data(dataset_path)
