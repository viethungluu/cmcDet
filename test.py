import os
import numpy as np
import argparse
import math

from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

import lightning as L
import albumentations as A

from models.backbone_utils import _dual_resnet_fpn_extractor
from models.resnet import CMCResNets
from models.cmc_retinanet import RetinaNetModule
from dataset.datamodule import PascalDataModule
from dataset.pascal.pascal_utils import generate_pascal_category_names
from dataset.colorspace_transforms import RGB2Lab

import warnings
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

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
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Name of dataset')
    parser.add_argument('--trainable-backbone-layers', type=int, default=0,
                        help='Number of trainable backbone layers.')
    parser.add_argument('--cmc-backbone', type=str, default='resnet50v2', 
                        choices=["resnet50v2", "resnet50v3"],
                        help='Backbone type')
    parser.add_argument('--v2', action='store_true')
    parser.add_argument('--seed', type=int, default=28,
                        help='Random seed')
    
    args = parser.parse_args()
    return args

def handle_test(args):
    # seed so that results are reproducible
    L.seed_everything(args.seed)

    if args.dataset_name == "vehicle":
        labels = ['big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck', 'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
    else:
        labels = ['person']
    labels = np.array(labels)
    num_classes = len(labels) + 1

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
    
    if args.backbone_choice == "dual":
        cmc = CMCResNets(name=args.cmc_backbone)
        
        if args.v2 and args.cmc_backbone == "resnet50v2":
            extra_blocks = LastLevelP6P7(4096, 256)
        elif args.v2 and args.cmc_backbone == "resnet50v3":
            extra_blocks = LastLevelP6P7(8192, 256)
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
        if args.v2:
            model = retinanet_resnet50_fpn_v2()
        else:        
            model = retinanet_resnet50_fpn()
        
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head = RetinaNetHead(in_channels, num_anchors, num_classes=num_classes)

    m = RetinaNetModule(model, classes=labels)

    # init trainer
    trainer = L.Trainer()

    # test (pass in the model)
    trainer.test(m,
                 dm,
                 ckpt_path=args.ckpt_path)

def main():
    args = _parse_args()
    handle_test(args)

if __name__ == "__main__":
    main()