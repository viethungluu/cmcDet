import os
import numpy as np
import argparse
import math

import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, retinanet_resnet50_fpn
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import albumentations as A

from models.backbone_utils import _dual_resnet_fpn_extractor
from models.resnet import CMCResNets
from models.cmc_retinanet import RetinaNetModule
from dataset.datamodule import PascalDataModule
from dataset.pascal.pascal_utils import generate_pascal_category_names
from dataset.colorspace_transforms import RGB2Lab

def _parse_args():
    parser = argparse.ArgumentParser(description="Training (CMC)RetinaNet on Pascal VOC format")
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset in Pascal VOC format')
    parser.add_argument('--backbone-choice', type=str, default='dual', 
                        choices=["dual", "single"],
                        help='Backbone type')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Path/URL of the checkpoint from which training is resumed')
    parser.add_argument('--test-batch-size', type=int, default=4,
                        help='Test/valid batch size')
    parser.add_argument('--seed', type=int, default=28,
                        help='Random seed')
    
    args = parser.parse_args()
    return args

def handle_test(args):
    # seed so that results are reproducible
    L.seed_everything(args.seed)

    model = RetinaNetModule.load_from_checkpoint(
        checkpoint_path=args.ckpt_path
    )

    if args.backbone_choice == "dual":
        test_transforms = A.Compose([
                                RGB2Lab(),
                            ],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        test_transforms = None

    dm = PascalDataModule(dataset_path=args.dataset_path,
                          test_batch_size=args.test_batch_size,
                          test_transforms=test_transforms,
                          seed=args.seed)
    dm.setup(stage="test")
    label_map = generate_pascal_category_names(dm.test_df)
    print(label_map)

    # init trainer
    trainer = L.Trainer()

    # test (pass in the model)
    trainer.test(model, dm)

def main():
    args = _parse_args()
    handle_test(args)

if __name__ == "__main__":
    main()