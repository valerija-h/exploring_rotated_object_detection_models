import os
import torch
import torchvision
import time
from torch.utils.data import DataLoader
from utils.cornell_dataset import CornellDataset
from utils import horizontal_transforms as T
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np

torch.manual_seed(0) # set seed for reproducibility

'''
TODO -
- create evaluation() function - checks if highest grasp is IOU > 0.25 and has an angle diff of 30 degrees
- add more datasets (multi_object and Jacquard are gold standards)
'''

# set the device used to train the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DATASET_PATH = 'dataset/cornell/RGB'
DATASET_PATH = 'dataset/cornell/RGD'
MODEL_PATH = 'models/test_02_cornell.pth'

# data preprocessing parameters
TEST_SPLIT = 0.20  # percentage of test samples from all samples
VAL_SPLIT = 0.10  # percentage of validation samples from training samples
SEED_SPLIT = 42

def val_one_epoch(model, data_loader, epoch, val_hist):
    model.train()  # keep it on train mode
    prog_bar = tqdm(data_loader, total=len(data_loader))

    val_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
    for i, data in enumerate(prog_bar):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # update loss for each batch
        prog_bar.set_description(desc=f"Validation Loss: {loss_value:.4f}")

        # store losses for history and reporting results
        val_hist[epoch]['loss'].append(loss_value)
        val_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
        val_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
        val_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
        val_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
    return val_hist

def train_one_epoch(model, data_loader, optimizer, epoch, train_hist):
    model.train()  # set model to train mode
    prog_bar = tqdm(data_loader, total=len(data_loader))

    train_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
    for i, data in enumerate(prog_bar):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # zero out gradients
        optimizer.zero_grad()

        # get losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # back propagation and adjust learning weights
        losses.backward()
        optimizer.step()

        # update loss for each batch
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        # store losses for history and reporting results
        train_hist[epoch]['loss'].append(loss_value)
        train_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
        train_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
        train_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
        train_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
    return train_hist

def create_model(freeze_layers=False):
    num_classes = len(class_mappings) + 1  # rotation classes + invalid proposal

    # load a model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # load a model pre-trained on COCO with cutom anchors
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,),)
    aspect_ratios = ((0.25, 0.5, 1.0),) * len(anchor_sizes)
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101', pretrained=True)
    model = torchvision.models.detection.FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator)      

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # create our own head models
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def print_losses(loss_hist, epoch, prefix=''):
    line = f'Epoch {epoch}'
    for name, value in loss_hist[epoch].items():
        line += f' - {prefix}{name}: {sum(value)/len(value):.3f}'
    print(line)

def train_network(train_data_loader, val_data_loader):
    # get pre-trained model
    model = create_model()

    # set model to device for training
    model.to(DEVICE)

    # set training parameters
    epochs = 20
    learning_rate = 0.0001
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # start training loop
    print("[INFO] training model...")
    train_hist, val_hist = {}, {}
    for e in range(epochs):
        print(f'Epoch {e}/{epochs - 1}')
        start = time.time()
        train_hist = train_one_epoch(model, train_data_loader, optimizer, e, train_hist)
        val_hist = val_one_epoch(model, val_data_loader, e, val_hist)
        print_losses(train_hist, e)
        print_losses(val_hist, e, prefix='val_')
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {e}")
    
    torch.save(model.state_dict(), f"{MODEL_PATH}") # save model

def evaluate_network(test_data_loader, visualize=False):
    # load model
    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # set model to evaluation mode
    model.eval()

    device = torch.device("cpu")

    for i, data in enumerate(test_data_loader):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        x_test = list(image.to(DEVICE) for image in images)
        y_test = [{k: v.to(device) for k, v in t.items()} for t in targets]

        y_pred = model(x_test)
        y_pred = [{k: v.to(device) for k, v in t.items()} for t in y_pred]

        if visualize == True:
            plot_prediction(images[0], y_pred[0], class_mappings)
            print(y_pred)
            #print(i)
            #print(y_pred)
            
        

'''
Plots a single image and its predictions.
'''
def plot_prediction(image, y_pred, class_mapping):
    # transform image back to PIL format for plotting
    image = torchvision.transforms.ToPILImage()(image)

    fig, ax = plt.subplots()
    ax.imshow(image)
    for b, (xmin, ymin, xmax, ymax) in enumerate(y_pred['boxes']):
        if b == 0:
        #if y_pred['scores'][b] > 0.5:
            w, h = (xmax.item() - xmin.item()), (ymax.item() - ymin.item())
            x, y = (xmax.item() - (w/ 2)), (ymax.item() - (h / 2))
            t_range = class_mapping[y_pred['labels'][b].item()]  # range of theta values [min_t, max_t]
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

# get data transforms
def get_transforms(class_mappings):
    return torchvision.transforms.Compose([
        #T.RandomShift(),
        #T.RandomRotate(class_mappings),
        T.CustomCrop(100, 160, 315),
        #T.RandomHorizontalFlip(class_mappings),
        #T.RandomVerticalFlip(class_mappings),
        T.ToTensor()
    ])


def get_data_loaders(train_dataset, test_dataset, val_dataset):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=T.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=T.collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=T.collate_fn
    )

    return train_loader, test_loader, val_loader


# split the dataset into training, testing and validation sets
def split_dataset(dataset):
    test_size = round(TEST_SPLIT * len(dataset))
    train_size = len(dataset) - test_size
    val_size = round(VAL_SPLIT * train_size)
    train_size = train_size - val_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size],
                                                                             generator=torch.Generator().manual_seed(SEED_SPLIT))
    return train_dataset, test_dataset, val_dataset

if __name__ == '__main__':
    # get dataset object and class mappings
    dataset = CornellDataset(DATASET_PATH)
    class_mappings = dataset.get_class_mapping()
    dataset.set_transforms(transforms=get_transforms(class_mappings))

    # split dataset into training and testing
    train_dataset, test_dataset, val_dataset = split_dataset(dataset)
    # get data loaders
    train_loader, test_loader, val_loader = get_data_loaders(train_dataset, test_dataset, val_dataset)
    print(f'[INFO] train dataset has {len(train_dataset)} samples.')
    print(f'[INFO] talidation dataset has {len(val_dataset)} samples.')
    print(f'[INFO] test dataset has {len(test_dataset)} samples.')

    # if the model_path exists - evaluate it, otherwise train a new model and save it to model_path
    if not os.path.exists(MODEL_PATH):
        train_network(train_loader, val_loader)

    # evaluate model
    evaluate_network(test_loader, visualize=True)

    # visualize a transformed example
    # T.visualise_transforms(test_loader, class_mappings)
