import numpy as np
import argparse

import torchvision
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

import lightning as L
import albumentations as A

import cv2

from models.backbone_utils import _dual_resnet_fpn_extractor
from models.resnet import CMCResNets
from models.cmc_retinanet import RetinaNetModule
from dataset.colorspace_transforms import RGB2Lab

def _parse_args():
    parser = argparse.ArgumentParser(description="Inference on image")
    parser.add_argument('-i', '--input-image', type=str, default=None,
                        help='Path/URL of the checkpoint from which training is resumed')
    parser.add_argument('-o', '--output-image', type=str, default=None,
                        help='Path/URL of the checkpoint from which training is resumed')
    parser.add_argument('--dataset', type=str, required=True, choices=["vehicle", "person"],
                        help='Name of dataset')
    parser.add_argument('--backbone-choice', type=str, default='dual', 
                        choices=["dual", "single"],
                        help='Backbone type')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Path/URL of the checkpoint from which training is resumed')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classess')
    parser.add_argument('--score-threshold', type=float, default=0.6,
                        help='Score threshold')
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

    if args.dataset == "vehicle":
        labels = ['__background__', 'big bus', 'big truck', 'bus-l-', 'bus-s-', 'car', 'mid truck', 'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
    else:
        labels = ['__background__', 'person']
    labels = np.array(labels)

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
            trainable_layers=0, 
            returned_layers=[2, 3, 4], 
            extra_blocks=extra_blocks
        )

        image_mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        image_std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        model = RetinaNet(backbone,
                          num_classes=args.num_classes,
                          image_mean=image_mean,
                          image_std=image_std)
    else:
        if args.v2:
            model = retinanet_resnet50_fpn_v2()
        else:        
            model = retinanet_resnet50_fpn()
        
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head = RetinaNetHead(in_channels, num_anchors, num_classes=args.num_classes)

    m = RetinaNetModule.load_from_checkpoint(args.ckpt_path, model=model)
    m.eval()

    transform = A.Compose([RGB2Lab()],
                            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    to_tensor = torchvision.transforms.ToTensor()
    
    image = cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2RGB)
    transformed = transform(image=image, bboxes=[], class_labels=[])
    image = transformed["image"]
    image = to_tensor(image)
    image = image.unsqueeze(0)
    image = image.to(device='cuda')

    preds = m(image.float())
    output = preds[0]
    vis = read_image(args.input_image)
    result = draw_bounding_boxes(vis, 
                                 boxes=output['boxes'][output['scores'] > args.score_threshold], 
                                 labels=labels[output['class_labels']],
                                 width=2)
    result = result.detach()
    result = F.to_pil_image(result)
    result.save(args.output_image)

def main():
    args = _parse_args()
    handle_test(args)

if __name__ == "__main__":
    main()