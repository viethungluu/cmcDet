import os
import numpy as np
import argparse
from PIL import Image

import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from models.cmc_retinanet import CMCRetinaNet
from dataset.datamodule import PascalDataModule

def _parse_args():
    parser = argparse.ArgumentParser(description="Training CMCRetinaNet on Pascal VOC format")
    # model parameters
    parser.add_argument('--cmc-backbone', type=str, default='resnet50v1', 
                        choices=["resnet50v1", "resnet50v2"],
                        help='Model type')
    parser.add_argument('--cmc-weights-path', type=str, default=None,
                        help='Path to CMC checkpoint')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset in Pascal VOC format')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the checkpoints and logs')
    parser.add_argument('--train-batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        help='Test/valid batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=28,
                        help='Random seed')
    
    # parser.add_argument('--rgb2bgr', action='store_true')

    args = parser.parse_args()
    return args

def handle_train(args):
    # loading data
    dm = PascalDataModule(dataset_path=args.dataset_path,
                          train_batch_size=args.train_batch_size,
                          test_batch_size=args.test_batch_size,
                          seed=args.seed)
    dm.setup(stage="fit")

    model = CMCRetinaNet(cmc_backbone=args.cmc_backbone,
                         cmc_weights_path=args.cmc_weights_path,
                         n_classes=2,
                         lr=args.lr)
    
    # Training
    trainer = L.Trainer(
        max_epochs=50,
        default_root_dir=args.save_path,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", min_delta=0.01, patience=3),
            ModelCheckpoint(dirpath=os.path.join(args.save_path, "checkpoints"),
                            save_top_k=1,
                            verbose=True,
                            monitor='val_loss',
                            mode='min',
                            filename='{epoch}-{val_loss:.2f}')
        ])

    trainer.fit(model, dm)

def main():
    args = _parse_args()
    handle_train(args)

if __name__ == "__main__":
    main()