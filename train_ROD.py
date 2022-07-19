import os.path as osp
import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import utils.transforms_ROD as TR

def train_model(cfg):
    # build the train dataset
    datasets = [build_dataset(cfg.data.train)]
    # build the rotated object detector model
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)  # train model

if __name__ == '__main__':
    model_choice = 's2anet'  # 'orcnn' 'r3det' or 's2anet'
    dataset_choice = 'cornell'

    # train model
    cfg = TR.get_ROD_config(model_choice, dataset_choice)
    train_model(cfg)
