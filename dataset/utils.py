import pandas as pd

def remove_invalid_annots(df: pd.DataFrame):
    df = df[df.xmax > df.xmin]
    df = df[df.ymax > df.ymin]
    df.reset_index(inplace=True, drop=True)
    return df