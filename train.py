import os
import numpy as np
import argparse
import math

import torch
import torch.nn as nn

import torchvision
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, retinanet_resnet50_fpn
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import albumentations as A

from models.backbone_utils import _dual_resnet_fpn_extractor
from models.resnet import CMCResNets
from models.cmc_retinanet import RetinaNetModule
from dataset.datamodule import PascalDataModule
from dataset.pascal.pascal_utils import generate_pascal_category_names
from dataset.colorspace_transforms import RGB2Lab

def _parse_args():
    parser = argparse.ArgumentParser(description="Training CMCRetinaNet on Pascal VOC format")
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset in Pascal VOC format')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the checkpoints and logs')
    parser.add_argument('--train-batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=4,
                        help='Test/valid batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=28,
                        help='Random seed')
    # model parameters
    subparsers = parser.add_subparsers(dest='backbone_choice', help='types of backbone model')
    s_parser = subparsers.add_parser("single", help="Single-Stream backbone")
    d_parser = subparsers.add_parser("dual", help="Dual-Stream backbone")   

    s_parser.add_argument('--resnet-backbone', type=str, default='resnet50', 
                        choices=["resnet50"],
                        help='Backbone type')
    s_parser.add_argument('--pretrained', action='store_true')
    s_parser.add_argument('--pretrained-backbone', action='store_true')

    d_parser.add_argument('--cmc-backbone', type=str, default='resnet50v2', 
                        choices=["resnet50v2"],
                        help='Backbone type')
    d_parser.add_argument('--cmc-weights-path', type=str, default=None,
                        help='Path to CMC checkpoint')
    d_parser.add_argument('--trainable-backbone-layers', type=int, default=0,
                        help='Number of trainable backbone layers.')

    args = parser.parse_args()
    return args

def handle_train(args):
    # seed so that results are reproducible
    L.seed_everything(args.seed)

    if args.backbone_choice == "dual":
        train_transforms = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                RGB2Lab(),
                            ], 
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        test_transforms = A.Compose([
                                RGB2Lab(),
                            ],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        train_transforms = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                            ],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        test_transforms = None

    dm = PascalDataModule(dataset_path=args.dataset_path,
                          train_batch_size=args.train_batch_size,
                          test_batch_size=args.test_batch_size,
                          train_transforms=train_transforms,
                          test_transforms=test_transforms,
                          seed=args.seed)
    dm.setup(stage="fit")
    label_map = generate_pascal_category_names(dm.train_df)
    num_classes = len(label_map)

    if args.backbone_choice == "dual":
        cmc = CMCResNets(name=args.cmc_backbone)
        if args.mc_weights_path:
            ckpt = torch.load(args.cmc_weights_path)
            cmc.load_state_dict(ckpt['model'])
            args.backbone_choice = "dual+"

        backbone = _dual_resnet_fpn_extractor(
            backbone_l=cmc.encoder.module.l_to_ab, 
            backbone_ab=cmc.encoder.module.ab_to_l, 
            trainable_layers=args.trainable_backbone_layers, 
            returned_layers=[2, 3, 4], 
            extra_blocks=LastLevelP6P7(256, 256)
        )

        image_mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        image_std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        model = RetinaNet(backbone,
                          num_classes=num_classes,
                          image_mean=image_mean,
                          image_std=image_std)
    else:        
        model = retinanet_resnet50_fpn(
                            pretrained=args.pretrained,
                            pretrained_backbone=args.pretrained_backbone,
                            num_classes=91 if args.pretrained else num_classes,
                            )

        if args.pretrained:
            # replace classification layer 
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head = RetinaNetHead(in_channels, num_anchors, num_classes=num_classes)
    
    m = RetinaNetModule(model, lr=args.lr)
    
    # Training
    trainer = L.Trainer(
        max_epochs=50,
        default_root_dir=args.save_path,
        callbacks=[
            EarlyStopping(monitor="map", mode="max", min_delta=0.01, patience=3),
            ModelCheckpoint(dirpath=os.path.join(args.save_path, "checkpoints", args.backbone_choice),
                            save_top_k=1,
                            verbose=True,
                            monitor='map',
                            mode='max',
                            filename='{epoch}-{map:.3f}')
        ])

    trainer.fit(m, dm)

def main():
    args = _parse_args()
    handle_train(args)

if __name__ == "__main__":
    main()