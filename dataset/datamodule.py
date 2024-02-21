import lightning as L

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.pascal.pascal_utils import convert_annotations_to_df, PascalDataset

def remove_invalid_annots(df):
    df = df[df.xmax > df.xmin]
    df = df[df.ymax > df.ymin]
    df.reset_index(inplace=True, drop=True)
    return df

class PascalDataModule(L.LightningDataModule):
    def __init__(self,
                 train_image_path=None,
                 val_image_path=None,
                 test_image_path=None,
                 train_annot_path=None,
                 val_annot_path=None,
                 test_annot_path=None,
                 train_batch_size=16,
                 test_batch_size=8,
                 seed=28):
        super().__init__()

        self.train_image_path = train_image_path,
        self.val_image_path = val_image_path,
        self.test_image_path = test_image_path,
        self.train_annot_path = train_annot_path,
        self.val_annot_path = val_annot_path,
        self.test_annot_path = test_annot_path,
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed

    def prepare_data(self):
        if self.train_image_path and self.train_annot_path:
            self.train_df = convert_annotations_to_df(self.train_annot_path, self.train_image_path, image_set="train")
            self.train_df = remove_invalid_annots(self.train_df)
        
        if self.val_image_path and self.val_annot_path:
            self.val_df = convert_annotations_to_df(self.val_annot_path, self.val_image_path, image_set="test")
            self.val_df = remove_invalid_annots(self.val_df)
        
        if self.test_image_path and self.test_annot_path:
            self.test_df  = convert_annotations_to_df(self.test_annot_path, self.test_image_path, image_set="test")
            self.test_df = remove_invalid_annots(self.test_df)

    def setup(self, stage):
        # TODO: update transform operations
        # according to  CMC pre-training
        transform = [transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        train_transform = transforms.Compose(transform + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAutocontrast(p=0.5),
        ])
        test_transform = transforms.Compose(transform)

        if stage == "fit" or stage is None:
            self.train_dataset  = PascalDataset(self.train_df, transforms=train_transform)
            self.val_dataset    = PascalDataset(self.val_df, transforms=test_transform)
        else:
            self.test_dataset    = PascalDataset(self.test_df, transforms=test_transform)

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset,
                              batch_size=self.train_batch_size,
                              shuffle=True,
                              num_workers=2)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2)

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2)