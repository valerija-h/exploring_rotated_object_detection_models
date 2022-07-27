import os
import torchvision
import time
from tqdm.auto import tqdm
from utils import transforms as T
from utils.cornell_dataset import CornellDataset
from utils.ocid_dataset import OCIDDataset
from config import *

torch.manual_seed(0)  # set seeds to ensure reproducibility


def train_network(dataset, model_name):
    """ This function trains a baseline model and saves it in the models directory.
       :param dataset: (Dataset) the dataset to train the baseline model on "ocid" or "cornell".
       :param model_name: (str) the path to save the model to.
    """
    train_loader, test_loader, val_loader = T.get_data_loaders(dataset)  # get data
    model = create_model(class_mappings)  # create a model with pre-trained weights
    model.to(TRAIN_DEVICE)  # set model to GPU for training

    # set training parameters
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # start training loop
    print(f"[INFO] Training a new baseline model...", flush=True)
    train_hist, val_hist = {}, {}  # to store training and validation losses
    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs}')
        start = time.time()
        train_hist = train_one_epoch(model, train_loader, optimizer, e, train_hist)
        val_hist = val_one_epoch(model, val_loader, e, val_hist)
        print_losses(train_hist, e)
        print_losses(val_hist, e, prefix='val_')
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {e+1}")
    # save model
    print(f"[INFO] Training Finished. Saving the model.")
    torch.save(model.state_dict(), f"{model_name}")

def create_model(class_mappings, freeze_layers=False):
    """ Returns a FasterRCNN model with pre-trained weights. """
    num_classes = len(class_mappings) + 1  # rotation classes + invalid proposal

    # load a model pre-trained on COCO with custom anchors
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,),)
    aspect_ratios = ((0.25, 0.5, 1.0),) * len(anchor_sizes)
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101', pretrained=True)
    model = torchvision.models.detection.FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator)

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # create our own head models
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, data_loader, optimizer, epoch, train_hist):
    model.train()  # set model to train mode
    prog_bar = tqdm(data_loader, total=len(data_loader))

    train_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
    for i, data in enumerate(prog_bar):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        images = list(image.to(TRAIN_DEVICE) for image in images)
        targets = [{k: v.to(TRAIN_DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()   # zero out gradients

        # get losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # back propagation and adjust learning weights
        losses.backward()
        optimizer.step()

        # update loss for each batch
        prog_bar.set_description(desc=f"Training Loss: {loss_value:.4f}")

        # store losses for history and reporting results
        train_hist[epoch]['loss'].append(loss_value)
        train_hist[epoch]['cls_loss'].append(loss_dict['loss_classifier'].item())
        train_hist[epoch]['bbox_loss'].append(loss_dict['loss_box_reg'].item())
        train_hist[epoch]['rpn_cls_loss'].append(loss_dict['loss_objectness'].item())
        train_hist[epoch]['rpn_bbox_loss'].append(loss_dict['loss_rpn_box_reg'].item())
    return train_hist


def val_one_epoch(model, data_loader, epoch, val_hist):
    model.train()  # keep it on train mode
    prog_bar = tqdm(data_loader, total=len(data_loader))

    val_hist[epoch] = {'loss': [], 'cls_loss': [], 'bbox_loss': [], 'rpn_cls_loss': [], 'rpn_bbox_loss': []}
    for i, data in enumerate(prog_bar):
        # get images and targets and send to device (i.e. GPU)
        images, targets = data
        images = list(image.to(TRAIN_DEVICE) for image in images)
        targets = [{k: v.to(TRAIN_DEVICE) for k, v in t.items()} for t in targets]

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


def print_losses(loss_hist, epoch, prefix=''):
    """ Prints out the losses of the RPN head and FasterRCNN head in the baseline model. """
    line = f'Epoch {epoch+1}'
    for name, value in loss_hist[epoch].items():
        line += f' - {prefix}{name}: {sum(value)/len(value):.3f}'
    print(line)


if __name__ == '__main__':
    dataset_choice = 'cornell'

    # get the data
    if dataset_choice == "cornell":
        dataset = CornellDataset(CORNELL_PATH, img_format=IMG_FORMAT)
    elif dataset_choice == "ocid":
        dataset = OCIDDataset(OCID_PATH, img_format=IMG_FORMAT)
    # get class mappings and transformations
    class_mappings = dataset.get_class_mapping()
    dataset.set_transforms(transforms=T.get_transforms(dataset_choice, class_mappings))

    # train model
    model_name = os.path.join(MODELS_PATH, f'baseline_{dataset_choice}_epoch_{EPOCHS}.pth')
    train_network(dataset, model_name)

