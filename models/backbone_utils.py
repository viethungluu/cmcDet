import warnings
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.models import resnet
from torchvision.models._api import _get_enum_from_fn, WeightsEnum
from torchvision.models._utils import handle_legacy_interface, IntermediateLayerGetter

from collections import OrderedDict

class DualBackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone_l: nn.Module,
        backbone_ab: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body_l = IntermediateLayerGetter(backbone_l, return_layers=return_layers)
        self.body_ab = IntermediateLayerGetter(backbone_ab, return_layers=return_layers)

        # hard-code the number of returned layer [2, 3, 4]
        # self.conv1 = nn.Conv2d(in_channels_list[0] * 2, in_channels_list[0], kernel_size=1, stride=1, padding="same")
        # self.conv2 = nn.Conv2d(in_channels_list[1] * 2, in_channels_list[1], kernel_size=1, stride=1, padding="same")
        # self.conv3 = nn.Conv2d(in_channels_list[2] * 2, in_channels_list[2], kernel_size=1, stride=1, padding="same")
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        l, ab = torch.split(x, [1, 2], dim=1)
        x_l = self.body_l(l)
        x_ab = self.body_ab(ab)
        
        x = OrderedDict()
        for i, k in enumerate(x_l.keys()):
            x[k] = torch.cat((x_l[k], x_ab[k]), dim=1)
            # if i == 0:
            #     x[k] = self.conv1(x[k])
            # elif i == 1:
            #     x[k] = self.conv2(x[k])
            # elif i == 2:
            #     x[k] = self.conv3(x[k])
        
        x = self.fpn(x)
        return x


def _dual_resnet_fpn_extractor(
    backbone_l: resnet.ResNet,
    backbone_ab: resnet.ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> DualBackboneWithFPN:

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
        
    for name, parameter in backbone_l.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    
    for name, parameter in backbone_ab.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    if cmc_backbone.endswith("v3"):
        in_channels_stage2 = backbone_l.inplanes // 8 * 2
    else:
        in_channels_stage2 = backbone_l.inplanes // 8

    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return DualBackboneWithFPN(
        backbone_l, backbone_ab, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )