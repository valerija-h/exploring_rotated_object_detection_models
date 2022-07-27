from mmdet.apis import inference_detector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import os
import torchvision
from torch.utils.data import DataLoader
from utils.cornell_dataset import CornellDataset
from utils.ocid_dataset import OCIDDataset
import utils.transforms as T
from shapely.geometry import Polygon
from config import *
from train_baseline import create_model
import sys
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility


def calc_grasp_metric(bbox_pred, theta_pred, y_test):
    plt.clf()
    for i in range(len(y_test['boxes'])):
        gt_class = y_test['labels'][i].item()
        gt_theta = (class_mappings[gt_class][0] + class_mappings[gt_class][1]) / 2
        gt_bbox = y_test['boxes'][i]

        # check if theta is within 30 degrees (0.523599 radians)
        if np.abs(gt_theta - theta_pred) < 0.523599 or (np.abs(np.abs(gt_theta - theta_pred) - np.pi)) < 0.523599:
            # now check if IOU > 0.25
            gt_grasp = Polygon(get_points(gt_bbox, gt_theta))
            pred_grasp = Polygon(get_points(bbox_pred, theta_pred))
            intersection = gt_grasp.intersection(pred_grasp).area / gt_grasp.union(pred_grasp).area
            if intersection > 0.25:
                return True
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=TRAIN_DEVICE))
    model.eval()
    return model

def evaluate_baseline_model():
    # set necessary transforms for baseline model
    dataset.set_transforms(T.get_transforms(dataset_choice, class_mappings))
    # load baseline model and checkpoint
    baseline_model = create_model(class_mappings).to(TRAIN_DEVICE)
    current_model = load_baseline_checkpoint(baseline_model, f"{MODELS_PATH}/baseline_" + dataset_choice + "_epoch_5.pth")
    sample_n = round(TEST_SPLIT * len(dataset))
    # get the test dataloader
    _, test_loader, _ = T.get_data_loaders(dataset)  # get data

    total_successes, total_time, total_confidence, total_pred = 0, 0, 0, 0
    for i, data in enumerate(test_loader):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        x_test = list(image.to(TRAIN_DEVICE) for image in images)
        y_test = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

        # calculate the time it takes to inference a sample
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y_pred = current_model(x_test)
        end.record()
        torch.cuda.synchronize()
        total_time += (start.elapsed_time(end) / 1000)  # compute time in seconds

        y_pred = [{k: v.to('cpu') for k, v in t.items()} for t in y_pred]

        # to make things easier - we will only work with a BS of 1 from now (i.e evaluate one sample img at a time)
        y_test, y_pred, x_test, image = y_test[0], y_pred[0], x_test[0], images[0]
        total_pred += (y_pred['scores'] > 0.3).sum()
        # get the best grasp prediction and its rotation class + theta value
        bbox_pred = y_pred['boxes'][0]
        class_pred = y_pred['labels'][0].item()
        theta_pred = (class_mappings[class_pred][0] + class_mappings[class_pred][1]) / 2
        is_correct = calc_grasp_metric(bbox_pred, theta_pred, y_test)  # check if predicted grasp is correct
        if is_correct:
            total_successes += 1
            total_confidence += y_pred['scores'][0].item()
    grasp_success_rate = (total_successes / sample_n) * 100
    print(
      f"[INFO] The baseline model got {total_successes}/{sample_n} ({grasp_success_rate:.2f}%) correct "
      f"predictions on the {dataset_choice} grasping dataset with an average FPS rate of {1/(total_time/sample_n):.2f}."
      f"\n On average the model predicted {(total_pred/sample_n):.2f} grasps per image with a confidence > 0.3 with the "
      f"most confident grasp in each image having an average confidence score of {(total_confidence/total_successes):.2f}.")
    results['baseline'] = grasp_success_rate
#
#
# def load_rotated_model(cfg, model_path):
#     cfg.model.pretrained = None
#     model = build_detector(cfg.model)
#     checkpoint = load_checkpoint(model, model_path, map_location=DEVICE)
#     model.CLASSES = checkpoint['meta']['CLASSES']
#     model.cfg = cfg
#     model.to(DEVICE)  # convert the model to GPU
#     model.eval()  # convert the model into evaluation mode
#     return model
#
#
# # def load_rotated_model_checkpoint(model, checkpoint):
# #     load_checkpoint(model, checkpoint, map_location=DEVICE)
#
# def evaluate_rotated_models():
#     cornell_dataset.set_transforms(get_evaluation_transforms("cornell", False))
#     ocid_dataset.set_transforms(get_evaluation_transforms("ocid", False))
#
#     for d, dataset_name in enumerate(results.keys()):
#         # load rotated models and their configs
#         orcnn_cfg = create_config_orcnn(
#             'libraries/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py',
#             checkpoint_file="/home/rpgcps/r01vh21/Documents/models/finals/orcnn_" + dataset_name + "_epoch_5.pth")
#         orcnn = load_rotated_model(orcnn_cfg,
#                                    "/home/rpgcps/r01vh21/Documents/models/finals/orcnn_" + dataset_name + "_epoch_5.pth")
#
#         s2anet_cfg = create_config_s2anet('libraries/mmrotate/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py',
#                                           checkpoint_file="/home/rpgcps/r01vh21/Documents/models/finals/s2anet_" + dataset_name + "_epoch_5.pth")
#         s2anet = load_rotated_model(s2anet_cfg,
#                                     "/home/rpgcps/r01vh21/Documents/models/finals/s2anet_" + dataset_name + "_epoch_5.pth")
#
#         redet_cfg = create_config_redet('libraries/mmrotate/configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py',
#                                         checkpoint_file="/home/rpgcps/r01vh21/Documents/models/finals/redet_" + dataset_name + "_epoch_5.pth")
#         redet = load_rotated_model(redet_cfg,
#                                    "/home/rpgcps/r01vh21/Documents/models/finals/redet_" + dataset_name + "_epoch_5.pth")
#
#         # get test data_loader
#         current_dataset = split_into_test_dataset(datasets[d])
#         data_loader = get_test_dataloader(current_dataset)
#
#         temp_results = {'orcnn': {'total_correct': 0, 'total_time': 0, 'total_confidence': 0, 'total_pred': 0},
#                         's2anet': {'total_correct': 0, 'total_time': 0, 'total_confidence': 0, 'total_pred': 0},
#                         'redet': {'total_correct': 0, 'total_time': 0, 'total_confidence': 0, 'total_pred': 0}}
#         models = [orcnn, s2anet, redet]
#
#         for i, data in enumerate(data_loader):
#             images, targets = data
#             img_path = datasets[d].get_img_path(targets[0]['image_id'].item())  # get image_path w.r.t non-DOTA dataset
#             img_path = '/home/rpgcps/r01vh21/Documents/DOTA_Datasets/' + dataset_name + '_DOTA/images/' + os.path.basename(
#                 img_path).replace("r.png", ".png")
#
#             # assess each model separately
#             for m, name in enumerate(temp_results.keys()):
#                 model = models[m]
#                 start = torch.cuda.Event(enable_timing=True)
#                 end = torch.cuda.Event(enable_timing=True)
#                 start.record()
#                 y_pred = inference_detector(model, img_path)
#                 end.record()
#                 torch.cuda.synchronize()
#                 inf_time = (start.elapsed_time(end))
#                 y_pred = sorted([x.tolist() for arr in y_pred for x in arr], key=lambda l: l[5], reverse=True)
#                 temp_results[name]['total_pred'] += len([p[5] for p in y_pred if p[5] > 0.3])
#                 y_pred = y_pred[0]  # take most confident prediction
#                 bbox_pred = [y_pred[0] - (y_pred[2] / 2), y_pred[1] - (y_pred[3] / 2),
#                              y_pred[0] + (y_pred[2] / 2), y_pred[1] + (y_pred[3] / 2)]
#                 # temp_results[name]['total_confidence'] += y_pred[5]
#                 is_correct = calc_grasp_metric(bbox_pred, y_pred[4], targets[0],
#                                                images[0])  # check if predicted grasp is correct
#                 if is_correct:
#                     temp_results[name]['total_correct'] += 1
#                     temp_results[name]['total_confidence'] += y_pred[5]
#                 temp_results[name]['total_time'] += (inf_time / 1000)
#
#         for m, name in enumerate(temp_results.keys()):
#             grasp_success_rate = (temp_results[name]['total_correct'] / len(current_dataset)) * 100
#             print(
#                 f"[INFO] The {name} rotated model got {temp_results[name]['total_correct']}/{len(current_dataset)} ({grasp_success_rate:.2f}%) correct "
#                 f"predictions on the {dataset_name} grasping dataset with an average FPS rate of {1 / ((temp_results[name]['total_time'] / len(current_dataset))):.2f}. "
#                 f"\n On average the model predicted {(temp_results[name]['total_pred'] / len(current_dataset)):.2f} grasps per image with a confidence > 0.3 with the most confident grasp in "
#                 f"each image having an average confidence score of {(temp_results[name]['total_confidence'] / temp_results[name]['total_correct']):.2f}.")
#         # results[dataset_name].append({'baseline': grasp_success_rate})
#

if __name__ == '__main__':
    dataset_choice = 'cornell'

    # get the data and class mappings
    if dataset_choice == "cornell":
        dataset = CornellDataset(CORNELL_PATH, img_format=IMG_FORMAT)
    elif dataset_choice == "ocid":
        dataset = OCIDDataset(OCID_PATH, img_format=IMG_FORMAT)
    else:
        sys.exit(f"[ERROR] '{dataset_choice}' is not a valid dataset choice. Please choose 'cornell' or 'ocid'.")
    class_mappings = dataset.get_class_mapping()

    results = {}  # to store the final results of each model
    evaluate_baseline_model()








