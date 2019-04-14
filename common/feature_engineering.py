import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def FE(df):
    # 简单线性组合
    df['new_feature'] = df['feature_1'] / df['feature_1']

    # 简单聚合
    df = pd.DataFrame({'id': ['Falcon', 'Falcon', 'Parrot', 'Parrot'], 'Max Speed': [380., 370., 24., 26.],
                       'wight': [1., 2., 3., 4.]})
    num_aggregations = {
        'Max Speed': ['min', 'max', 'mean', 'var']
    }
    count_aggregations = {
        'wight': ['count']
    }
    # 按照上面定义的num_aggregations,count_aggregations两种聚合方式进行聚合
    agg = df.groupby('id').agg({**num_aggregations, **count_aggregations})
    df = df.join(agg, how='left', on='id')

    return df


