import glob
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
encoder = LabelEncoder()


def convert_annotations_to_df(data_dir, image_set="train"):
    xml_list = []
    for xml_file in glob.glob(data_dir + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            bbx = member.find("bndbox")
            xmin = int(bbx.find("xmin").text)
            ymin = int(bbx.find("ymin").text)
            xmax = int(bbx.find("xmax").text)
            ymax = int(bbx.find("ymax").text)
            label = member.find("name").text

            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                label,
                xmin,
                ymin,
                xmax,
                ymax,
            )
            xml_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df["filename"] = [
        os.path.join(data_dir, xml_df["filename"][i]) for i in range(len(xml_df))
    ]

    if image_set == "train":
        # label encoder encodes the labels from 0
        # we need to add +1 so that labels are encode from 1 as our
        # model reserves 0 for background class.
        xml_df["labels"] = encoder.fit_transform(xml_df["class"]) + 1
    elif image_set == "val" or image_set == "test":
        xml_df["labels"] = encoder.transform(xml_df["class"]) + 1
    return xml_df

def generate_pascal_category_names(df: pd.DataFrame):
    """
    Genearte a List contatining the Category names, 
    The index of the List will correspond to the integer
    value of the category name

    Args:
        df: pd.Dataframe: A pandas dataframe in the `pascal_voc_csv format`
    """
    cs = df["class"].unique()
    ls = df["labels"].unique()

    CATEGORY_NAMES = list(np.zeros(len(ls) + 1))
    for i, x in enumerate(ls):
        CATEGORY_NAMES[x] = cs[i]

    # Add the background class to the Category names
    # Since the labels start from the 1, we set the 0 value to be the
    # background class
    CATEGORY_NAMES[0] = "__background__"
    return CATEGORY_NAMES

