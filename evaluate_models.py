from mmdet.apis import inference_detector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import os
from utils.cornell_dataset import CornellDataset
from utils.ocid_dataset import OCIDDataset
import utils.transforms as T
import utils.transforms_ROD as TR
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
    sample_n = round(TEST_SPLIT * len(dataset))
    # get the test dataloader
    _, test_loader, _ = T.get_data_loaders(dataset)  # get data
    # load baseline model and checkpoint
    baseline_model = create_model(class_mappings).to(TRAIN_DEVICE)
    current_model = load_baseline_checkpoint(baseline_model, f"{MODELS_PATH}/baseline_{dataset_choice}_epoch_5.pth")

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

def load_rotated_model(model_choice, dataset_choice):
    # get the config file and build model
    cfg = TR.get_ROD_config(model_choice, dataset_choice)
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, f"{MODELS_PATH}/{model_choice}_{dataset_choice}_epoch_5.pth",
                                 map_location=TRAIN_DEVICE)  # load the checkpoint
    model.CLASSES = checkpoint['meta']['CLASSES']  # assign classes
    model.cfg = cfg
    # convert model to GPU and set to evaluation mode
    model.to(TRAIN_DEVICE)
    model.eval()
    return model

def evaluate_rotated_models():
    dataset.set_transforms(T.get_transforms(dataset_choice, class_mappings, tt=False))
    sample_n = round(TEST_SPLIT * len(dataset))
    # get the test dataloader
    _, test_loader, _ = T.get_data_loaders(dataset)  # get data

    # load rotated models and their configs
    model_names = ['orcnn', 's2anet', 'redet']
    models, model_results = [], {}
    for m in model_names:
        models.append(load_rotated_model(m, dataset_choice))
        model_results[m] = {'total_correct': 0, 'total_time': 0, 'total_confidence': 0, 'total_pred': 0}

    for i, data in enumerate(test_loader):
        images, targets = data
        img_path = dataset.get_img_path(targets[0]['image_id'].item())  # get image_path w.r.t non-DOTA dataset
        img_path = os.path.join(DOTA_PATH, dataset_choice, "images", os.path.basename(img_path).replace("r.png", ".png"))

        # assess each model separately
        for m, name in enumerate(model_names):
            model = models[m]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y_pred = inference_detector(model, img_path)
            end.record()
            torch.cuda.synchronize()
            inf_time = (start.elapsed_time(end))

            y_pred = sorted([x.tolist() for arr in y_pred for x in arr], key=lambda l: l[5], reverse=True)
            model_results[name]['total_pred'] += len([p[5] for p in y_pred if p[5] > 0.3])
            y_pred = y_pred[0]  # take most confident prediction
            bbox_pred = [y_pred[0] - (y_pred[2] / 2), y_pred[1] - (y_pred[3] / 2),
                         y_pred[0] + (y_pred[2] / 2), y_pred[1] + (y_pred[3] / 2)]

            is_correct = calc_grasp_metric(bbox_pred, y_pred[4], targets[0])  # check if predicted grasp is correct
            if is_correct:
                model_results[name]['total_correct'] += 1
                model_results[name]['total_confidence'] += y_pred[5]
            model_results[name]['total_time'] += (inf_time / 1000)

    for m, name in enumerate(model_names):
        grasp_success_rate = (model_results[name]['total_correct'] / sample_n) * 100
        print(
          f"[INFO] The {name} rotated model got {model_results[name]['total_correct']}/{sample_n} "
          f"({grasp_success_rate:.2f}%) correct predictions on the {dataset_choice} grasping dataset with an average "
          f"FPS rate of {1 / ((model_results[name]['total_time'] / sample_n)):.2f}. On average the model predicted "
          f"{(model_results[name]['total_pred'] / sample_n):.2f} grasps per image with a confidence > 0.3 with the most "
          f"confident grasp in each image having an average confidence score of "
          f"{(model_results[name]['total_confidence'] / model_results[name]['total_correct']):.2f}.")
        results[name] = grasp_success_rate


if __name__ == '__main__':
    dataset_choice = 'ocid'

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
    evaluate_rotated_models()









