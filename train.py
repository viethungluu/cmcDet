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
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the checkpoints and logs')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Path/URL of the checkpoint from which training is resumed')
    parser.add_argument('--train-batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=4,
                        help='Test/valid batch size')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Max epochs')
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1,
                        help='Run val loop every 10 training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr-decay', action='store_true')
    parser.add_argument('--mpt', help="Enable Mixed Precision Training", action='store_true')
    parser.add_argument('--seed', type=int, default=28,
                        help='Random seed')
    parser.add_argument('--trainable-backbone-layers', type=int, default=3,
                        help='Number of trainable backbone layers.')
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
                        choices=["resnet50v2", "resnet50v3"],
                        help='Backbone type')
    d_parser.add_argument('--cmc-weights-path', type=str, default=None,
                        help='Path to CMC checkpoint')

    args = parser.parse_args()
    return args

def handle_train(args):
    # seed so that results are reproducible
    L.seed_everything(args.seed)

    if args.backbone_choice == "dual":
        train_transforms = A.Compose([
                                A.Rotate(limit=15),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.GaussNoise(),
                                RGB2Lab(),
                            ], 
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        test_transforms = A.Compose([
                                RGB2Lab(),
                            ],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        train_transforms = A.Compose([
                                A.Rotate(limit=15),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.GaussNoise(),
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
    print(label_map)
    num_classes = len(label_map)

    if args.backbone_choice == "dual":
        if args.ckpt_path is not None:
            args.mc_weights_path = None
        
        cmc = CMCResNets(name=args.cmc_backbone)
        if args.cmc_weights_path:
            ckpt = torch.load(args.cmc_weights_path)
            cmc.load_state_dict(ckpt['model'])
            args.backbone_choice = "dual+"

        if args.cmc_backbone.endswith("v3"):
            extra_blocks = LastLevelP6P7(2048, 256)
        else:
            extra_blocks = LastLevelP6P7(256, 256)

        backbone = _dual_resnet_fpn_extractor(
            backbone_l=cmc.encoder.module.l_to_ab, 
            backbone_ab=cmc.encoder.module.ab_to_l, 
            trainable_layers=args.trainable_backbone_layers, 
            returned_layers=[2, 3, 4], 
            extra_blocks=extra_blocks
        )

        image_mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        image_std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        model = RetinaNet(backbone,
                          num_classes=num_classes,
                          image_mean=image_mean,
                          image_std=image_std)
    else:
        if args.ckpt_path is not None:
            args.pretrained = False
            args.pretrained_backbone = False

        model = retinanet_resnet50_fpn(
                            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1 if args.pretrained else None,
                            weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if args.pretrained_backbone else None,
                            trainable_backbone_layers=args.trainable_backbone_layers, 
                            num_classes=91 if args.pretrained else num_classes,
                            )

        if args.pretrained:
            # replace classification layer 
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head = RetinaNetHead(in_channels, num_anchors, num_classes=num_classes)
    
    m = RetinaNetModule(model, 
                        lr=args.lr,
                        lr_decay=args.lr_decay)
    
    # Training
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        default_root_dir=args.save_path,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            # EarlyStopping(monitor="map", mode="max", min_delta=0.01, patience=15),
            ModelCheckpoint(dirpath=os.path.join(args.save_path, "checkpoints", args.backbone_choice),
                            save_top_k=1,
                            verbose=True,
                            monitor='map',
                            mode='max',
                            filename='{epoch}-{map:.3f}')
        ])

    kwarg = {}
    if args.ckpt_path:
        kwarg["ckpt_path"] = args.ckpt_path
    if args.mpt:
        kwarg["precision"] = 16
    
    trainer.fit(m, 
                dm,
                **kwarg)

def main():
    args = _parse_args()
    handle_train(args)

if __name__ == "__main__":
    main()