import os

import lightning as L

from torch.utils.data import DataLoader
import albumentations as A

from dataset.pascal.pascal_dataset import PascalDataset
from dataset.pascal.pascal_utils import convert_annotations_to_df
from dataset.utils import remove_invalid_annots, clip_invalid_annots, collate_fn
from dataset.colorspace_transforms import RGB2Lab

class PascalDataModule(L.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 train_batch_size=8,
                 test_batch_size=4,
                 seed=28):
        super().__init__()

        self.train_dir = os.path.join(dataset_path, "train")
        self.val_dir   = os.path.join(dataset_path, "valid")
        self.test_dir  = os.path.join(dataset_path, "test")
        
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed

    def setup(self, stage):
        if os.path.isdir(self.train_dir):
            self.train_df = convert_annotations_to_df(self.train_dir, image_set="train")
            self.train_df = remove_invalid_annots(self.train_df)
        
        if os.path.isdir(self.val_dir):
            self.val_df = convert_annotations_to_df(self.val_dir, image_set="test")
            self.val_df = remove_invalid_annots(self.val_df)
        
        if os.path.isdir(self.test_dir):
            self.test_df    = convert_annotations_to_df(self.test_dir, image_set="test")
            self.test_df    = remove_invalid_annots(self.test_df)

        colorspace_transform = RGB2Lab()

        # ToTensor will be performed inside dataset
        # normalize will be performed inside RetinaNet
        # We will only do color transform to specific color space here
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RGB2Lab()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        test_transforms = A.Compose([
            RGB2Lab()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        if stage == "fit" or stage is None:
            self.train_dataset  = PascalDataset(self.train_df, transforms=train_transforms)
            self.val_dataset    = PascalDataset(self.val_df, transforms=test_transforms)
        else:
            self.test_dataset   = PascalDataset(self.test_df, transforms=test_transforms)

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset,
                              batch_size=self.train_batch_size,
                              shuffle=True,
                              num_workers=2,
                              collate_fn= collate_fn)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2,
                              collate_fn= collate_fn)

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2,
                              collate_fn= collate_fn)