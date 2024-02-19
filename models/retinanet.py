import torch, torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
import lightning as L

from resnet import CMCResNets

class CMCRetinaNet(nn.Module):
    def __init__(self, 
                 backbone: CMCResNets,
                 trainable_backbone_layers: int=0):
        super(CMCRetinaNet, self).__init__()
        
        self.l_to_ab = backbone.l_to_ab
        self.l_to_ab = _resnet_fpn_extractor(
                            self.l_to_ab, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))

        self.ab_to_l = backbone.ab_to_l
        self.ab_to_l = _resnet_fpn_extractor(
                            self.ab_to_l, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))

    def forward(self, x, layer=7):
        pass