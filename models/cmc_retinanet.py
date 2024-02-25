import torch
from torchvision.models.detection.retinanet import RetinaNet
from torchmetrics.detection import MeanAveragePrecision

import lightning as L

class RetinaNetModule(L.LightningModule):
    def __init__(self,
                 model,
                 lr: float=1e-3,
                 lr_decay: bool=False):
        super().__init__()

        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.metric = MeanAveragePrecision(iou_type="bbox")

    def forward(self, x):
        # return loss_dict in fit stage
        # return preds in inference (equal to model.eval())
        return self.model(x)

    def configure_optimizers(self):
        # select parameters that require gradient calculation
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min"
        )

        if self.lr_decay:
            return {
                "optimizer": optimizer, 
                "lr_scheduler": scheduler, 
                "monitor": "train_loss"
            }

        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.float() for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) 
        self.log("train_loss", losses, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images))
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.float() for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets

        preds = self.model(images, targets)
        self.metric.update(preds, targets)
        map_dict = self.metric.compute()
        
        log_dict = {
            'map' : map_dict['map'],
            'map_50': map_dict['map_50'],
            'map_75': map_dict['map_75'],
            'map_small': map_dict['map_small'],
            'map_medium': map_dict['map_medium'],
            'map_large': map_dict['map_large'],
            'mar_1': map_dict['mar_1'],
            'mar_10': map_dict['mar_10'],
            'mar_100': map_dict['mar_100'],
            'mar_small': map_dict['mar_small'],
            'mar_medium': map_dict['mar_medium'],
            'mar_large': map_dict['mar_large']
        }

        self.log(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images))
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.float() for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets

        preds = self.model(images, targets)
        self.metric.update(preds, targets)
        map_dict = self.metric.compute()
        log_dict = {
            'map' : map_dict['map'],
            'map_50': map_dict['map_50'],
            'map_75': map_dict['map_75'],
            'map_small': map_dict['map_small'],
            'map_medium': map_dict['map_medium'],
            'map_large': map_dict['map_large'],
            'mar_1': map_dict['mar_1'],
            'mar_10': map_dict['mar_10'],
            'mar_100': map_dict['mar_100'],
            'mar_small': map_dict['mar_small'],
            'mar_medium': map_dict['mar_medium'],
            'mar_large': map_dict['mar_large']
        }

        self.log(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images))