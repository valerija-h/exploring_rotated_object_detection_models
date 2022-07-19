import os
import torch
import torchvision
import time

from shapely.geometry import Polygon
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np

from utils import transforms as T
from config import *

torch.manual_seed(0)  # set seeds to ensure reproducibility


#
#
# def val_one_epoch(model, data_loader, epoch, val_hist):
#     model.train()  # keep it on train mode
#     prog_bar = tqdm(data_loader, total=len(data_loader))
#
#     val_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
#     for i, data in enumerate(prog_bar):
#         # get images and targets and send to device (i.e. GPU)
#         images, targets = data
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#
#         with torch.no_grad():
#             loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#
#         # update loss for each batch
#         prog_bar.set_description(desc=f"Validation Loss: {loss_value:.4f}")
#
#         # store losses for history and reporting results
#         val_hist[epoch]['loss'].append(loss_value)
#         val_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
#         val_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
#         val_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
#         val_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
#     return val_hist
#
# def train_one_epoch(model, data_loader, optimizer, epoch, train_hist):
#     model.train()  # set model to train mode
#     prog_bar = tqdm(data_loader, total=len(data_loader))
#
#     train_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
#     for i, data in enumerate(prog_bar):
#         # get images and targets and send to device (i.e. GPU)
#         images, targets = data
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#
#         # zero out gradients
#         optimizer.zero_grad()
#
#         # get losses
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#
#         # back propagation and adjust learning weights
#         losses.backward()
#         optimizer.step()
#
#         # update loss for each batch
#         prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
#
#         # store losses for history and reporting results
#         train_hist[epoch]['loss'].append(loss_value)
#         train_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
#         train_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
#         train_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
#         train_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
#     return train_hist
#
# def create_model(freeze_layers=False):
#     num_classes = len(class_mappings) + 1  # rotation classes + invalid proposal
#
#     # load a model pre-trained on COCO
#     # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#     # load a model pre-trained on COCO with cutom anchors
#     anchor_sizes = ((16,), (32,), (64,), (128,), (256,),)
#     aspect_ratios = ((0.25, 0.5, 1.0),) * len(anchor_sizes)
#     anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
#     backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101', pretrained=True)
#     model = torchvision.models.detection.FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator)
#
#     if freeze_layers:
#         for param in model.parameters():
#             param.requires_grad = False
#
#     # create our own head models
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
#     return model
#
# def print_losses(loss_hist, epoch, prefix=''):
#     line = f'Epoch {epoch}'
#     for name, value in loss_hist[epoch].items():
#         line += f' - {prefix}{name}: {sum(value)/len(value):.3f}'
#     print(line)
#
# def train_network(train_data_loader, val_data_loader):
#     # get pre-trained model
#     model = create_model()
#
#     # set model to device for training
#     model.to(DEVICE)
#
#     # set training parameters
#     epochs = 20
#     learning_rate = 0.0001
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.Adam(params, lr=learning_rate)
#
#     # start training loop
#     print("[INFO] training model...")
#     train_hist, val_hist = {}, {}
#     for e in range(epochs):
#         print(f'Epoch {e}/{epochs - 1}')
#         start = time.time()
#         train_hist = train_one_epoch(model, train_data_loader, optimizer, e, train_hist)
#         val_hist = val_one_epoch(model, val_data_loader, e, val_hist)
#         print_losses(train_hist, e)
#         print_losses(val_hist, e, prefix='val_')
#         end = time.time()
#         print(f"Took {((end - start) / 60):.3f} minutes for epoch {e}")
#
#     torch.save(model.state_dict(), f"{MODEL_PATH}") # save model
#
# def get_points(box, t):
#     xmin, ymin, xmax, ymax = box
#     w, h = xmax - xmin, ymax - ymin
#     x, y = xmax - (w / 2), ymax - (h / 2)
#
#     w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
#             h / 2) * np.cos(t)
#     bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
#     br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
#     return (tl_x, tl_y), (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y)
#
# def calc_grasp_metric(y_pred, y_test, img):
#     # get the best bbox pred, theta pred
#     bbox_pred = y_pred[0]['boxes'][0]
#     class_pred = y_pred[0]['labels'][0].item()
#     theta_pred = (class_mappings[class_pred][0] + class_mappings[class_pred][1]) / 2
#
#     for i in range(len(y_test[0]['boxes'])):
#         gt_class = y_test[0]['labels'][i].item()
#         gt_theta = (class_mappings[gt_class][0] + class_mappings[gt_class][1]) / 2
#         gt_bbox = y_test[0]['boxes'][i]
#
#         # check if theta is within 30 degrees (0.523599 radians)
#         if np.abs(gt_theta - theta_pred) < 0.523599 or (np.abs(np.abs(gt_theta - theta_pred) - np.pi)) < 0.523599:
#             # now check if IOU > 0.25
#             gt_grasp = Polygon(get_points(gt_bbox, gt_theta))
#             pred_grasp = Polygon(get_points(bbox_pred, theta_pred))
#
#             intersection = gt_grasp.intersection(pred_grasp).area / gt_grasp.union(pred_grasp).area
#             if intersection > 0.25:
#                 plt.imshow(torchvision.transforms.ToPILImage()(img))
#                 x, y = gt_grasp.exterior.xy
#                 plt.plot(x, y, 'b')
#                 x, y = pred_grasp.exterior.xy
#                 plt.plot(x, y, 'r')
#                 plt.title(f'IOU: {intersection}')
#                 plt.show()
#                 return True
#     return False  # if metrics aren't met
#
#
# def evaluate_network(test_data_loader, visualize=False):
#     # load model
#     model = create_model().to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     # set model to evaluation mode
#     model.eval()
#
#     device = torch.device("cpu")
#
#     for i, data in enumerate(test_data_loader):
#         # get images and targets and send to device (i.e. GPU)
#         images, targets = data
#         x_test = list(image.to(DEVICE) for image in images)
#         y_test = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         y_pred = model(x_test)
#         y_pred = [{k: v.to(device) for k, v in t.items()} for t in y_pred]
#
#         # warning this will only work with BS of 1 for now
#         is_correct = calc_grasp_metric(y_pred, y_test, images[0])
#
#         if visualize == True:
#             plot_prediction(images[0], y_pred[0], class_mappings)
#             #print(y_pred)
#             #print(i)
#             #print(y_pred)
#
#
#
# '''
# Plots a single image and its predictions.
# '''
# def plot_prediction(image, y_pred, class_mapping):
#     # transform image back to PIL format for plotting
#     image = torchvision.transforms.ToPILImage()(image)
#
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     for b, (xmin, ymin, xmax, ymax) in enumerate(y_pred['boxes']):
#         if b == 0:
#         #if y_pred['scores'][b] > 0.5:
#             w, h = (xmax.item() - xmin.item()), (ymax.item() - ymin.item())
#             x, y = (xmax.item() - (w/ 2)), (ymax.item() - (h / 2))
#             t_range = class_mapping[y_pred['labels'][b].item()]  # range of theta values [min_t, max_t]
#             t = (t_range[0] + t_range[1]) / 2
#             w_cos, w_sin, h_sin, h_cos = (w / 2) * np.cos(t), (w / 2) * np.sin(t), (h / 2) * np.sin(t), (
#                     h / 2) * np.cos(t)
#             bl_x, bl_y, tl_x, tl_y = x - w_cos + h_sin, y - w_sin - h_cos, x - w_cos - h_sin, y - w_sin + h_cos
#             br_x, br_y, tr_x, tr_y = x + w_cos + h_sin, y + w_sin - h_cos, x + w_cos - h_sin, y + w_sin + h_cos
#             plt.plot([bl_x, tl_x], [bl_y, tl_y], c='black')
#             plt.plot([br_x, tr_x], [br_y, tr_y], c='black')
#             rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none',
#                                      transform=Affine2D().rotate_around(*(x, y), t) + ax.transData)
#             ax.add_patch(rect)
#
#     plt.show()


if __name__ == '__main__':
    dataset_choice = 'cornell'
    train_loader, test_loader, val_loader = T.get_data_loaders(dataset_choice)

    # # if the model_path exists - evaluate it, otherwise train a new model and save it to model_path
    # if not os.path.exists(MODEL_PATH):
    #     train_network(train_loader, val_loader)
    #
    # # evaluate model
    # evaluate_network(test_loader, visualize=True)

    #visualize a transformed example
    #T.visualise_transforms(test_loader, class_mappings)
