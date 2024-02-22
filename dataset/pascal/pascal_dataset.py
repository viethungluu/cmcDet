import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import cv2

class PascalDataset(Dataset):
    """
    Creates a object detection Dataset instance.

    The dataset `__getitem__` should return:
      - image: a Tensor of size `(channels, H, W)`
      - target: a dict containing the following fields
        * `boxes (FloatTensor[N, 4])`: the coordinates of the N bounding boxes in `[x0, y0, x1, y1]` format, 
                                       ranging from 0 to W and 0 to H
        * `labels (Int64Tensor[N])`: the label for each bounding box. 0 represents always the background class.
        * `image_id (Int64Tensor[1])`: an image identifier. It should be unique between all the images in the dataset, 
                                       and is used during evaluation
        * `area (Tensor[N])`: The area of the bounding box. This is used during evaluation with the COCO metric, 
                              to separate the metric scores between small, medium and large boxes.
        * `iscrowd (UInt8Tensor[N])`: instances with iscrowd=True will be ignored during evaluation.
      - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, 
                                   and is used during evaluation.
    Args:
        1. dataframe : A pd.Dataframe instance or str corresponding to the 
                       path to the dataframe.
        For the Dataframe the `filename` column should correspond to the path to the images.
        Each row to should one annotations in the the form `xmin`, `ymin`, `xmax`, `yman`.
        Labels should be integers in the `labels` column.
        To convert the pascal voc data in csv format use the `get_pascal` function.
        
        2. transforms: (A.Compose) transforms should be a albumentation transformations.
                        the bbox params should be set to `pascal_voc` & to pass in class
                        use `class_labels`                  
    """

    def __init__(self, dataframe, transforms):
        if isinstance(dataframe, str):
            dataframe = pd.read_csv(dataframe)

        self.transforms = transforms
        self.df = dataframe
        self.image_ids = self.df["filename"].unique()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        # Grab the Image
        image_id = self.image_ids[index]
        image = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)

        # extract the bounding boxes
        records = self.df[self.df["filename"] == image_id]
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values

        # claculate area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Grab the Class Labels
        class_labels = records["labels"].values.tolist()

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        # apply transformations
        # transformed = self.transforms(image=im, bboxes=boxes, class_labels=class_labels)
        # image = transformed["image"]
        # boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        # class_labels = torch.tensor(transformed["class_labels"])
        if self.transforms:
            image, boxes = self.transforms(image, np.array(boxes))
        
        # target dictionary
        target = {}
        target['boxes'] = torch.as_tensor(boxes)
        target["labels"] = torch.as_tensor(class_labels)
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_idx = torch.tensor([index])
        target["image_id"] = image_idx

        return image, target, image_idx