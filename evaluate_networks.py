from mmrotate.datasets.builder import ROTATED_DATASETS
from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.datasets.dota import DOTADataset
from mmcv import Config
import os.path as osp
import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
import matplotlib.pyplot as plt
from mmcv.runner import load_checkpoint
import os
import torch
import torchvision
import time
from torch.utils.data import DataLoader
from utils.cornell_dataset import CornellDataset
from utils.ocid_dataset import OCIDDataset
from utils.jacquard_dataset import JacquardDataset
from horizontal_network import create_model, plot_prediction
from shapely.geometry import Polygon

from utils import horizontal_transforms as T
#import torchvision.transforms as TT

# from rotated_network import create_config
# from rotated_network_s2_2 import create_config
# from rotated_network_r3 import create_config
from config import create_config_orcnn, create_config_s2anet, create_config_redet

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility

TEST_SPLIT = 0.20  # percentage of test samples from all samples
VAL_SPLIT = 0.10  # percentage of validation samples from training samples
SEED_SPLIT = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_grasp_metric(bbox_pred, theta_pred, y_test, image=None):
    plt.clf() 
    for i in range(len(y_test['boxes'])):
        gt_class = y_test['labels'][i].item()
        gt_theta = (class_mappings[gt_class][0] + class_mappings[gt_class][1]) / 2
        gt_bbox = y_test['boxes'][i]

        # gt_grasp = Polygon(get_points(gt_bbox, gt_theta))
        # x, y = gt_grasp.exterior.xy
        # plt.plot(x, y, 'b')
        # check if theta is within 30 degrees (0.523599 radians)
        if np.abs(gt_theta - theta_pred) < 0.523599 or (np.abs(np.abs(gt_theta - theta_pred) - np.pi)) < 0.523599:
            # now check if IOU > 0.25
            gt_grasp = Polygon(get_points(gt_bbox, gt_theta))
            pred_grasp = Polygon(get_points(bbox_pred, theta_pred))

            # gt_grasp = Polygon(get_points(gt_bbox, gt_theta))
            #pred_grasp = Polygon(get_points(bbox_pred, theta_pred))
            # x, y = gt_grasp.exterior.xy
            # plt.plot(x, y, 'black')
            # x, y = pred_grasp.exterior.xy
            # plt.plot(x, y, 'yellow')
            # plt.imshow(image)
            # print(gt_theta, theta_pred)
            # plt.show()

            intersection = gt_grasp.intersection(pred_grasp).area / gt_grasp.union(pred_grasp).area
            if intersection > 0.25:
                return True
    
    # gt_grasp = Polygon(get_points(gt_bbox, gt_theta))
    # pred_grasp = Polygon(get_points(bbox_pred, theta_pred))
    # # x, y = gt_grasp.exterior.xy
    # # plt.plot(x, y, 'b')
    # x, y = pred_grasp.exterior.xy
    # plt.plot(x, y, 'r')
    # plt.imshow(image)
    # plt.show()

    return False


def get_points(bbox, t):
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    x, y = xmax - (w / 2), ymax - (h / 2)

    w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
            h / 2) * np.cos(t)
    bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
    br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
    return (tl_x, tl_y), (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y)


def load_baseline_checkpoint(model, checkpoint_path):
    # load model and set to evaluation mode
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

def get_evaluation_transforms(crop=None, to_tensor=True):
    transforms = []
    # custom cropping for OCID or Cornell dataset - to remove background noise
    if crop == 'ocid':
        transforms.append(T.CustomCrop(10, 10, (590, 460)))
    elif crop == 'cornell':
        transforms.append(T.CustomCrop(100, 160, 315))
    if to_tensor == True:
        transforms.append(T.ToTensor())
    return torchvision.transforms.Compose(transforms)


def split_into_test_dataset(dataset):
    test_size = round(TEST_SPLIT * len(dataset))
    train_size = len(dataset) - test_size
    val_size = round(VAL_SPLIT * train_size)
    train_size = train_size - val_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 SEED_SPLIT))
    return test_dataset

def get_test_dataloader(dataset):
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=T.collate_fn
    )
    return test_loader

def evaluate_baseline_model():
    # set necessary transforms for baseline model
    cornell_dataset.set_transforms(get_evaluation_transforms("cornell", True))
    ocid_dataset.set_transforms(get_evaluation_transforms("ocid", True))
    
    # load baseline model
    baseline_model = create_model(class_mappings).to(DEVICE)
    for d, dataset_name in enumerate(results.keys()):
        # load checkpoint from corresponding dataset
        current_model = load_baseline_checkpoint(baseline_model, "/home/rpgcps/r01vh21/Documents/models/finals/baseline_" + dataset_name + "_epoch_5.pth")
        # load necessary transformations for baseline model
        current_dataset = split_into_test_dataset(datasets[d])
        # get test data loader
        data_loader = get_test_dataloader(current_dataset)

        total_successes, total_time, total_confidence, total_pred = 0, 0, 0, 0
        for i, data in enumerate(data_loader):
            # get images and targets and send to device (i.e. GPU)
            images, targets = data
            x_test = list(image.to(DEVICE) for image in images)
            y_test = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y_pred = current_model(x_test)
            end.record()
            torch.cuda.synchronize()
            total_time += (start.elapsed_time(end)/1000) 

            y_pred = [{k: v.to('cpu') for k, v in t.items()} for t in y_pred]

            # to make things easier - we will only work with a BS of 1 from now (i.e evaluate one grasp at a time)
            y_test, y_pred, x_test, image = y_test[0], y_pred[0], x_test[0], images[0]
            # get the best bbox pred, theta pred
            #total_confidence += y_pred['scores'][0].item()
            total_pred += (y_pred['scores'] > 0.3).sum()
            bbox_pred = y_pred['boxes'][0]
            class_pred = y_pred['labels'][0].item()
            theta_pred = (class_mappings[class_pred][0] + class_mappings[class_pred][1]) / 2
            is_correct = calc_grasp_metric(bbox_pred, theta_pred, y_test, image)  # check if predicted grasp is correct
            if is_correct:
                total_successes += 1
                total_confidence += y_pred['scores'][0].item()
        grasp_success_rate = (total_successes / len(current_dataset)) * 100
        print(f"[INFO] The baseline model got {total_successes}/{len(current_dataset)} ({grasp_success_rate:.2f}%) correct "
              f"predictions on the {dataset_name} grasping dataset with an average FPS rate of {1/((total_time/len(current_dataset))):.2f}."
              f"\n On average the model predicted {(total_pred/len(current_dataset)):.2f} grasps per image with a confidence > 0.3 with the most confident grasp in each image having an average confidence score of {(total_confidence/total_successes):.2f}.")
        results[dataset_name].append({'baseline': grasp_success_rate})


def load_rotated_model(cfg, model_path):
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, model_path, map_location=DEVICE)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = cfg
    model.to(DEVICE)  # convert the model to GPU
    model.eval()  # convert the model into evaluation mode
    return model

# def load_rotated_model_checkpoint(model, checkpoint):
#     load_checkpoint(model, checkpoint, map_location=DEVICE)

def evaluate_rotated_models():
    cornell_dataset.set_transforms(get_evaluation_transforms("cornell", False))
    ocid_dataset.set_transforms(get_evaluation_transforms("ocid", False))

    for d, dataset_name in enumerate(results.keys()):
        # load rotated models and their configs
        orcnn_cfg = create_config_orcnn('libraries/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py', 
                            checkpoint_file="/home/rpgcps/r01vh21/Documents/models/finals/orcnn_" + dataset_name + "_epoch_5.pth")
        orcnn = load_rotated_model(orcnn_cfg, "/home/rpgcps/r01vh21/Documents/models/finals/orcnn_" + dataset_name + "_epoch_5.pth")

        s2anet_cfg = create_config_s2anet('libraries/mmrotate/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py', 
                            checkpoint_file="/home/rpgcps/r01vh21/Documents/models/finals/s2anet_" + dataset_name + "_epoch_5.pth")
        s2anet = load_rotated_model(s2anet_cfg, "/home/rpgcps/r01vh21/Documents/models/finals/s2anet_" + dataset_name + "_epoch_5.pth")

        redet_cfg = create_config_redet('libraries/mmrotate/configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py', 
                            checkpoint_file="/home/rpgcps/r01vh21/Documents/models/finals/redet_" + dataset_name + "_epoch_5.pth")
        redet = load_rotated_model(redet_cfg, "/home/rpgcps/r01vh21/Documents/models/finals/redet_" + dataset_name + "_epoch_5.pth")

        # get test data_loader
        current_dataset = split_into_test_dataset(datasets[d])
        data_loader = get_test_dataloader(current_dataset)

        temp_results = {'orcnn':    {'total_correct':0, 'total_time':0, 'total_confidence':0, 'total_pred':0}, 
                        's2anet':   {'total_correct':0, 'total_time':0, 'total_confidence':0, 'total_pred':0}, 
                        'redet':    {'total_correct':0, 'total_time':0, 'total_confidence':0, 'total_pred':0}}
        models = [orcnn, s2anet, redet]

        for i, data in enumerate(data_loader):
            images, targets = data
            img_path = datasets[d].get_img_path(targets[0]['image_id'].item()) # get image_path w.r.t non-DOTA dataset
            img_path = '/home/rpgcps/r01vh21/Documents/DOTA_Datasets/' + dataset_name + '_DOTA/images/' + os.path.basename(img_path).replace("r.png", ".png")
 
            # assess each model separately
            for m, name in enumerate(temp_results.keys()):
                model = models[m]
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                y_pred = inference_detector(model, img_path)
                end.record()
                torch.cuda.synchronize()
                inf_time = (start.elapsed_time(end)) 
                y_pred = sorted([x.tolist() for arr in y_pred for x in arr], key=lambda l:l[5], reverse=True)
                temp_results[name]['total_pred'] += len([p[5] for p in y_pred if p[5] > 0.3])
                y_pred = y_pred[0] # take most confident prediction  
                bbox_pred = [y_pred[0] - (y_pred[2]/2), y_pred[1] - (y_pred[3]/2),
                            y_pred[0] + (y_pred[2]/2), y_pred[1] + (y_pred[3]/2)]
                # temp_results[name]['total_confidence'] += y_pred[5]
                is_correct = calc_grasp_metric(bbox_pred, y_pred[4], targets[0], images[0])  # check if predicted grasp is correct
                if is_correct:
                    temp_results[name]['total_correct'] += 1
                    temp_results[name]['total_confidence'] += y_pred[5]
                temp_results[name]['total_time'] += (inf_time/1000)
            
        for m, name in enumerate(temp_results.keys()):
            grasp_success_rate = (temp_results[name]['total_correct'] / len(current_dataset)) * 100 
            print(f"[INFO] The {name} rotated model got {temp_results[name]['total_correct']}/{len(current_dataset)} ({grasp_success_rate:.2f}%) correct "
                  f"predictions on the {dataset_name} grasping dataset with an average FPS rate of {1/((temp_results[name]['total_time']/len(current_dataset))):.2f}. "
                  f"\n On average the model predicted {(temp_results[name]['total_pred']/len(current_dataset)):.2f} grasps per image with a confidence > 0.3 with the most confident grasp in " 
                  f"each image having an average confidence score of {(temp_results[name]['total_confidence']/temp_results[name]['total_correct']):.2f}.")
        # results[dataset_name].append({'baseline': grasp_success_rate})


if __name__ == '__main__':
    print("[INFO] Loading all datasets...")
    cornell_dataset = CornellDataset("dataset/cornell/RGD")
    #jacquard_dataset = JacquardDataset("dataset/jacquard", rgd=True)
    ocid_dataset = OCIDDataset("dataset/ocid", rgd=True)
    class_mappings = cornell_dataset.get_class_mapping()
    print("[INFO] All datasets have been loaded.")
    # datasets = [cornell_dataset, ocid_dataset]
    # results = {'cornell':[], 'ocid':[]}
    # datasets = [ocid_dataset]
    # results = {'ocid':[]}
    datasets = [cornell_dataset]
    results = {'cornell':[]}

    evaluate_rotated_models()
    evaluate_baseline_model()








