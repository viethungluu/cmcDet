import torch
from torchvision.models.detection.retinanet import RetinaNet
from torchmetrics.detection import MeanAveragePrecision

import lightning as L
import pl_bolts

import matplotlib.pyplot as plt

# from models.eval_utils import ConfusionMatrix

def plot_one_curve(ax, thr, x, y, title="", xlabel="Recall", ylabel="Precision", style="-"):
    try:
        _ = ax.plot(
            x,
            y,
            style,
            label="{0}".format(title),
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    except Exception as e:
        print(e)

class RetinaNetModule(L.LightningModule):
    def __init__(self,
                 model,
                 classes=None,
                 lr: float=1e-3,
                 eta_min: float=0,
                 lr_scheduler: str=None,
                 warmup_epochs: int=1,
                 max_epochs: int=100,
                 last_epoch:int=-1):
        super().__init__()

        self.model = model
        self.lr = lr
        self.eta_min = eta_min
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.last_epoch = last_epoch
        self.classes = classes

        self.metric = MeanAveragePrecision(iou_type="bbox", 
                                           backend='pycocotools', 
                                           extended_summary=True)
        # self.cm = ConfusionMatrix(len(self.classes), 0.5, 0.5)

    def forward(self, x):
        # return loss_dict in fit stage
        # return preds in inference (equal to model.eval())
        return self.model(x)

    def configure_optimizers(self):
        # select parameters that require gradient calculation
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)

        if self.lr_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.max_epochs, 
                last_epoch=self.last_epoch
            )
        elif self.lr_scheduler == "LinearWarmupCosineAnnealingLR":
            scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer, 
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                last_epoch=self.last_epoch
            )
        elif self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min"
            )
        elif self.lr_scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                eta_min=self.eta_min,
                last_epoch=self.last_epoch
            )
        else:
            scheduler = None

        if scheduler:
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
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images))
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.float() for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets

        preds = self.model(images, targets)
        self.metric.update(preds, targets)
         
    def on_validation_epoch_end(self):
        map_dict = self.metric.compute()
        self.log('map', map_dict['map'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_50', map_dict['map_50'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_75', map_dict['map_75'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_small', map_dict['map_small'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_medium', map_dict['map_medium'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_large', map_dict['map_large'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_1', map_dict['mar_1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_10', map_dict['mar_10'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_100', map_dict['mar_100'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_small', map_dict['mar_small'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_medium', map_dict['mar_medium'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_large', map_dict['mar_large'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.float() for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]  # Unpack the Targets

        preds = self.model(images, targets)
        self.metric.update(preds, targets)
        
        # self.cm.process_batch(preds, targets)
    
    def on_test_epoch_end(self):
        map_dict = self.metric.compute()
        self.log('map', map_dict['map'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_50', map_dict['map_50'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_75', map_dict['map_75'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_small', map_dict['map_small'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_medium', map_dict['map_medium'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('map_large', map_dict['map_large'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_1', map_dict['mar_1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_10', map_dict['mar_10'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_100', map_dict['mar_100'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_small', map_dict['mar_small'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_medium', map_dict['mar_medium'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mar_large', map_dict['mar_large'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metric.reset()

        # self.cm.print_matrix()

        # plot
        precision_s = map_dict["precision"]
        rec_thresholds = [i/100 for i in range(101)]  # this's defined by torchmetrics document by default
        # precision-recall curves for each classes
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.set_ylim(-0.0, 1.1)
        ax.set_xlim(-0.0, 1.1)
        for c, classname in enumerate(self.classes):
            thr = 0.5
            precision = precision_s[0,:, c, 0, -1]
            plot_one_curve(ax, thr, precision, rec_thresholds, title=classname)
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0)
        plt.title(f"Precision-Recall curves")
        plt.savefig("pr_curve.png")
