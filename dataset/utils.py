import pandas as pd

def remove_invalid_annots(df: pd.DataFrame):
    df = df[df.xmax > df.xmin]
    df = df[df.ymax > df.ymin]
    df.reset_index(inplace=True, drop=True)
    return df

def clip_invalid_annots(df: pd.DataFrame):
    cols = ['xmax', 'ymax', 'xmin', 'ymin']
    df[cols] = df[cols].clip(0, 640)
    return df

def collate_fn(batch):
    return tuple(zip(*batch))
