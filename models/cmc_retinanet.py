import torch
import torch.nn as nn

import lightning as L

from torchvision.models.detection.retinanet import RetinaNet
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from models.resnet import MyResNetsCMC
from models.backbone_utils import _dual_resnet_fpn_extractor

class CMCRetinaNet(L.LightningModule):
    def __init__(self,
                 cmc_backbone: str='resnet50v1',
                 cmc_weights_path: str=None,
                 trainable_backbone_layers: int=0,
                 n_classes: int=2,
                 lr: float=1e-3):
        super().__init__()

        cmc = MyResNetsCMC(name=cmc_backbone)
        if cmc_weights_path:
            ckpt = torch.load(cmc_weights_path)
            cmc.load_state_dict(ckpt['model'])
    
        backbone = _dual_resnet_fpn_extractor(
            backbone_l=cmc.encoder.l_to_ab, 
            backbone_ab=cmc.encoder.ab_to_l, 
            trainable_backbone_layers=trainable_backbone_layers, 
            returned_layers=[2, 3, 4], 
            extra_blocks=LastLevelP6P7(2048, 256)
        )

        self.model = RetinaNet(backbone, num_classes=n_classes)

        self.lr = lr

    def forward(self, x):
        # return loss_dict in fit stage
        # return preds in inference (equal to model.eval())
        return self.model(x)

    def configure_optimizers(self):
        # select parameters that require gradient calculation
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss_dict = self(inputs, labels)
        losses = sum(loss for loss in loss_dict.values()) 
        self.log("train_loss", losses, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss_dict = self(inputs, labels)
        losses = sum(loss for loss in loss_dict.values()) 

        self.log('val_loss', losses, on_step=False, on_epoch=True, prog_bar=True, logger=True)
