import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

# Preprocess application_train.csv and application_test.csv
def read_data(num_rows=None):
    # Read data and merge
    train_df = pd.read_csv('train.csv', nrows=num_rows)
    y = train_df['Survived']
    train_size = len(train_df)
    train_df.drop(['Survived'], axis=1)
    test_df = pd.read_csv('test.csv', nrows=num_rows)
    test_size = len(test_df)
    df = train_df.append(test_df).reset_index()
    del train_df, test_df
    gc.collect()
    return df, y, train_size, test_size

def EDA(train_df):

    train_df.describe()
    train_df.head(10)

    # 单变量分布 http://seaborn.pydata.org/tutorial/distributions.html#distribution-tutorial
    # 查看所有变量分布柱状图情况 n个小图
    train_df.hist(bins=50, figsize=(20, 15))
    plt.show()
    # 查看目标变量target分布情况
    sns.distplot(train_df.target)

    # 多变量关系
    # 多变量相关系数的热力图
    f, ax = plt.subplots(figsize=(25, 25))
    sns.heatmap(train_df.corr(), annot=True, linewidth=.5, fmt=".3f", ax=ax)
    plt.show()

    # 2变量：类别型 http://seaborn.pydata.org/tutorial/categorical.html#categorical-tutorial
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="max_floor", y="price_doc", data=train_df)
    plt.ylabel('Median Price', fontsize=12)
    plt.xlabel('Max Floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

    # 2变量：关系型 http://seaborn.pydata.org/tutorial/relational.html#relational-tutorial
    sns.relplot(x="feature_!", y="target", data=train_df)
    sns.scatterplot(x="feature_!", y="target", data=train_df)

# 要达到没有缺失值、异常值、int/float/time/text类型以外的数据
def preprocess_data(df, nan_as_category=False):
    # 把object类型的日期时间数据转化为datetime
    date = pd.to_datetime(df['timestamp'])
    df['date'] = df.date.dt.date
    df['month'] = df.date.dt.month
    df['week'] = df.date.dt.week
    df['day'] = df.date.dt.day

    # 类别型数据onehot
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['Sex']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # 处理缺失值
    df.fillna(df.mean(), inplace=True)  # 填充均值
    df.fillna(df.median(), inplace=True)  # 填充中位数
    df.fillna(0, inplace=True)  # 填充0

    # 离群点平滑
    ulimit = np.percentile(df.price_doc.values, 99.5)
    llimit = np.percentile(df.price_doc.values, 0.5)
    df['price_doc'].ix[df['price_doc'] > ulimit] = ulimit
    df['price_doc'].ix[df['price_doc'] < llimit] = llimit

    return df

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



