from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from mmcv import Config
from mmdet.apis import set_random_seed
import os
from config import *

@ROTATED_DATASETS.register_module()
class GraspDataset(DOTADataset):
    """Grasp dataset for detection."""
    CLASSES = ('grasp',)


#############################################################################
# ------------ CONFIGS FOR EACH ROTATED OBJECT DETECTOR ------------------- #
#############################################################################
def get_ROD_config(model_choice, dataset_choice, print_config=False):
    if model_choice == "orcnn":
        return create_orcnn_config(dataset_choice, print_config)
    elif model_choice == "s2anet":
        return create_s2anet_config(dataset_choice, print_config)
    elif model_choice == "redet":
        return create_redet_config(dataset_choice, print_config)
    elif model_choice == "r3det":
        return create_r3det_config(dataset_choice, print_config)

def create_orcnn_config(dataset_choice, print_config=False):
    # path to config file, checkpoint file, dataset directory and a directory to save model checkpoints
    cfg_file = os.path.join(MMROTATE_PATH, 'configs', 'oriented_rcnn', 'oriented_rcnn_r50_fpn_1x_dota_le90.py')
    chk_file = os.path.join(MMROTATE_PATH, 'checkpoints', 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth')
    data_path = os.path.join(DOTA_PATH, dataset_choice)
    print(os.path.join(DOTA_PATH, dataset_choice))
    model_dir = os.path.join(MODELS_PATH, f'orcnn_{dataset_choice}')

    img_prefix = 'images'
    data_type = 'GraspDataset'  # has to be same as register_module above
    cfg = Config.fromfile(cfg_file)  # load default config file

    '''--- CONFIG params related to data --- '''
    cfg.dataset_type = data_type
    cfg.data_root = data_path
    cfg.data.test.type = data_type
    cfg.data.test.data_root = data_path
    cfg.data.test.ann_file = 'test_labels'
    cfg.data.test.img_prefix = img_prefix
    cfg.data.train.type = data_type
    cfg.data.train.data_root = data_path
    cfg.data.train.ann_file = 'train_labels'
    cfg.data.train.img_prefix = img_prefix
    cfg.data.val.type = data_type
    cfg.data.val.data_root = data_path
    cfg.data.val.ann_file = 'val_labels'
    cfg.data.val.img_prefix = img_prefix
    ''' --- CONFIG params related to model --- '''
    cfg.model.roi_head.bbox_head.num_classes = 1  # we only have one class - 'grasp'
    cfg.model.rpn_head.anchor_generator.ratios = [0.25, 0.5, 1.0]  # make sure anchors are same
    cfg.load_from = chk_file
    ''' --- CONFIG params related to training ---  '''
    cfg.optimizer = dict(type='Adam', lr=0.0001)
    cfg.lr_config.warmup = None
    cfg.work_dir = model_dir  # to save files and logs
    cfg.runner.max_epochs = EPOCHS
    cfg.log_config.interval = 10
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 5  # reduce evaluation times
    cfg.checkpoint_config.interval = 5  # checkpoint saving interval to reduce storage cost
    # IMPORTANT - set seeds for reproducibility
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'
    # ADD DEFAULT DATA AUGMENTATIONS
    # -- but make sure random flip % is same as baseline
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25],
            direction=['horizontal', 'vertical'],
            version='le90'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    if print_config:
        print(f'Config:\n{cfg.pretty_text}')
    return cfg


def create_r3det_config(dataset_choice, print_config=False):
    # path to config file, checkpoint file, dataset directory and a directory to save model checkpoints
    cfg_file = os.path.join(MMROTATE_PATH, 'configs', 'r3det', 'r3det_r50_fpn_1x_dota_oc.py')
    chk_file = os.path.join(MMROTATE_PATH, 'checkpoints', 'r3det_r50_fpn_1x_dota_oc-b1fb045c.pth')
    data_path = os.path.join(DOTA_PATH, dataset_choice)
    model_dir = os.path.join(MODELS_PATH, f'r3det_{dataset_choice}')

    img_prefix = 'images'
    data_type = 'GraspDataset'  # has to be same as register_module above
    cfg = Config.fromfile(cfg_file)  # load default config file

    '''--- CONFIG params related to data --- '''
    cfg.dataset_type = data_type
    cfg.data_root = data_path
    cfg.data.test.type = data_type
    cfg.data.test.data_root = data_path
    cfg.data.test.ann_file = 'test_labels'
    cfg.data.test.img_prefix = img_prefix
    cfg.data.train.type = data_type
    cfg.data.train.data_root = data_path
    cfg.data.train.ann_file = 'train_labels'
    cfg.data.train.img_prefix = img_prefix
    cfg.data.val.type = data_type
    cfg.data.val.data_root = data_path
    cfg.data.val.ann_file = 'val_labels'
    cfg.data.val.img_prefix = img_prefix
    ''' --- CONFIG params related to model --- '''
    cfg.model.bbox_head.num_classes = 1  # we only have one class - 'grasp'
    cfg.model.refine_heads = [
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=1,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='oc',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ]  # we only have one class - 'grasp'
    cfg.model.bbox_head.anchor_generator.ratios = [0.25, 0.5, 1.0]  # make sure anchors are same
    cfg.load_from = chk_file
    ''' --- CONFIG params related to training ---  '''
    cfg.optimizer = dict(type='Adam', lr=0.0001)
    cfg.lr_config.warmup = None
    cfg.work_dir = model_dir  # to save files and logs
    cfg.runner.max_epochs = EPOCHS
    cfg.log_config.interval = 10
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 5  # reduce evaluation times
    cfg.checkpoint_config.interval = 5  # checkpoint saving interval to reduce storage cost
    # IMPORTANT - set seeds for reproducibility
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'
    # ADD DEFAULT DATA AUGMENTATIONS
    # -- but make sure random flip is same as baseline
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25],
            direction=['horizontal', 'vertical'],
            version='oc'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    if print_config:
        print(f'Config:\n{cfg.pretty_text}')
    return cfg


def create_s2anet_config(dataset_choice, print_config=False):
    # path to config file, checkpoint file, dataset directory and a directory to save model checkpoints
    cfg_file = os.path.join(MMROTATE_PATH, 'configs', 's2anet', 's2anet_r50_fpn_1x_dota_le135.py')
    chk_file = os.path.join(MMROTATE_PATH, 'checkpoints', 's2anet_r50_fpn_1x_dota_le135-5dfcf396.pth')
    data_path = os.path.join(DOTA_PATH, dataset_choice)
    model_dir = os.path.join(MODELS_PATH, f's2anet_{dataset_choice}')

    img_prefix = 'images'
    data_type = 'GraspDataset'  # has to be same as register_module above
    cfg = Config.fromfile(cfg_file)  # load default config file

    '''--- CONFIG params related to data --- '''
    cfg.dataset_type = data_type
    cfg.data_root = data_path
    cfg.data.test.type = data_type
    cfg.data.test.data_root = data_path
    cfg.data.test.ann_file = 'test_labels'
    cfg.data.test.img_prefix = img_prefix
    cfg.data.train.type = data_type
    cfg.data.train.data_root = data_path
    cfg.data.train.ann_file = 'train_labels'
    cfg.data.train.img_prefix = img_prefix
    cfg.data.val.type = data_type
    cfg.data.val.data_root = data_path
    cfg.data.val.ann_file = 'val_labels'
    cfg.data.val.img_prefix = img_prefix
    ''' --- CONFIG params related to model --- '''
    cfg.model.fam_head.num_classes = 1
    cfg.model.odm_head.num_classes = 1
    # cfg.model.fam_head.anchor_generator.ratios = [0.25, 0.5, 1.0]
    cfg.load_from = chk_file
    ''' --- CONFIG params related to training ---  '''
    cfg.optimizer = dict(type='Adam', lr=0.0001)
    cfg.lr_config.warmup = None
    cfg.work_dir = model_dir  # to save files and logs
    cfg.runner.max_epochs = EPOCHS
    cfg.log_config.interval = 10
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 5  # reduce evaluation times
    cfg.checkpoint_config.interval = 5  # checkpoint saving interval to reduce storage cost
    # IMPORTANT - set seeds for reproducibility
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'
    # ADD DEFAULT DATA AUGMENTATIONS
    # -- but make sure random flip is same as baseline
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25],
            direction=['horizontal', 'vertical'],
            version='le135'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    if print_config:
        print(f'Config:\n{cfg.pretty_text}')
    return cfg


def create_redet_config(dataset_choice, print_config=False):
    # path to config file, checkpoint file, dataset directory and a directory to save model checkpoints
    cfg_file = os.path.join(MMROTATE_PATH, 'configs', 'redet', 'redet_re50_refpn_1x_dota_ms_rr_le90.py')
    chk_file = os.path.join(MMROTATE_PATH, 'checkpoints', 'redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth')
    data_path = os.path.join(DOTA_PATH, dataset_choice)
    model_dir = os.path.join(MODELS_PATH, f'redet_{dataset_choice}')

    img_prefix = 'images'
    data_type = 'GraspDataset'  # has to be same as register_module above
    cfg = Config.fromfile(cfg_file)  # load default config file

    '''--- CONFIG params related to data --- '''
    cfg.dataset_type = data_type
    cfg.data_root = data_path
    cfg.data.test.type = data_type
    cfg.data.test.data_root = data_path
    cfg.data.test.ann_file = 'test_labels'
    cfg.data.test.img_prefix = img_prefix
    cfg.data.train.type = data_type
    cfg.data.train.data_root = data_path
    cfg.data.train.ann_file = 'train_labels'
    cfg.data.train.img_prefix = img_prefix
    cfg.data.val.type = data_type
    cfg.data.val.data_root = data_path
    cfg.data.val.ann_file = 'val_labels'
    cfg.data.val.img_prefix = img_prefix
    ''' --- CONFIG params related to model --- '''
    cfg.model.roi_head.bbox_head = [
        dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range='le90',
                    norm_factor=2,
                    edge_swap=True,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range='le90',
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ]  # we only have one class - 'grasp'
    cfg.model.rpn_head.anchor_generator.ratios = [0.25, 0.5, 1.0] # make sure anchor ratios are the same
    cfg.load_from = chk_file
    ''' --- CONFIG params related to training ---  '''
    cfg.optimizer = dict(type='Adam', lr=0.0001)
    cfg.lr_config.warmup = None
    cfg.work_dir = model_dir  # to save files and logs
    cfg.runner.max_epochs = EPOCHS
    cfg.log_config.interval = 10
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 5  # reduce evaluation times
    cfg.checkpoint_config.interval = 5  # checkpoint saving interval to reduce storage cost
    # IMPORTANT - set seeds for reproducibility
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'
    # ADD DEFAULT DATA AUGMENTATIONS
    # -- but make sure random flip is same as baseline
    angle_version = 'le90'
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25],
        direction=['horizontal', 'vertical'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]
    return cfg