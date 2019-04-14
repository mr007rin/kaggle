import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import pickle

from sklearn import preprocessing
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import VotingClassifier, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

orig_cols=all_cols=useful_cols=num_cols=cate_cols=txt_cols=todrop_cols=[]
target_col=''
all_data=pd.DataFrame()
y = pd.DataFrame()

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def save(name,object):
    time_suffix=datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
    pickle.dump(object+time_suffix, open(name, 'wb'))

def load(filepath):
    list = pickle.load(open(filepath, 'rb'))



# Preprocess application_train.csv and application_test.csv
def read_data(num_rows=None):
    # Read data and merge
    global target_col,num_cols,all_data
    train_df = pd.read_csv('D:/datasets/Titanic/train.csv', nrows=num_rows)
    target_col = 'Survived'
    int_cols = [col for col in train_df.columns if train_df[col].dtype == 'int64']
    float_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']
    num_cols=int_cols+float_cols
    test_df = pd.read_csv('D:/datasets/Titanic/test.csv', nrows=num_rows)
    orig_cols = list(test_df.columns)
    all_data = train_df.append(test_df).reset_index()
    del train_df, test_df
    gc.collect()


def EDA():
    global all_data
    all_data.describe()
    # 查看dataframe数据类型
    print(all_data.dtypes)
    # 查看目标变量的分布状况
    # 查看各个int float类型数据的相关性heatmap
    # 查看各个int float类型数据与目标变量之间的相关性heatmap

    # 查看缺失值情况
    total = all_data.isnull().sum().sort_values(ascending=False)
    percent = (all_data.isnull().sum() / len(all_data)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(nan_as_category=True):
    global cate_cols,all_data,target_col
    cate_cols = [col for col in all_data.columns if all_data[col].dtype == 'object']
    cate_cols.remove(target_col)
    all_data=pd.get_dummies(all_data, columns=cate_cols, dummy_na=nan_as_category)
    new_columns = [c for c in all_data.columns if c not in orig_cols]
    return new_columns

# 要达到没有缺失值、异常值、int/float/time/text类型以外的数据
def preprocess_data(nan_as_category=False):
    # 把object类型的日期时间数据转化为datetime
    global all_data,cate_cols
    date = pd.to_datetime(all_data['timestamp'])
    all_data['date'] = all_data.date.dt.date
    all_data['month'] = all_data.date.dt.month
    all_data['week'] = all_data.date.dt.week
    all_data['day'] = all_data.date.dt.day

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['Sex']:
        all_data[bin_feature], uniques = pd.factorize(all_data[bin_feature])
    # Categorical features with One-Hot encode
    all_data, cate_cols = one_hot_encoder(all_data, nan_as_category)

    # todo 要怎么处理缺失值
    # all_data.fillna(all_data.mean(), inplace=True)  # 填充均值
    all_data.fillna(all_data.median(), inplace=True)  # 填充中位数

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan 异常值处理
    # all_data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    save('clear.dat',all_data)


def feature_engineering():
    global all_data,useful_cols
    # 这是最简单的，几个变量之间的线性组合，直接当数值写就好了
    # all_data['PAYMENT_PERC'] = all_data['AMT_PAYMENT'] / all_data['AMT_INSTALMENT']
    # all_data['PAYMENT_DIFF'] = all_data['AMT_INSTALMENT'] - all_data['AMT_PAYMENT']

    # 通过sklearn标准化数据，如果只是基于树的模型可以省略这一步
    all_data = preprocessing.scale(all_data)


def rmsle_cv(model, train, y_train):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


def modelling_regression(x_train, y):
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    score = rmsle_cv(lasso, x_train, y)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(ENet, x_train, y)
    print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(KRR, x_train, y)
    print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(GBoost, x_train, y)
    print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(model_lgb, x_train, y)
    print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


def modelling_classification(x_train, y):
    kfold = StratifiedKFold(n_splits=10)
    random_state = 2
    # 确定评分标准
    scorer = 'accuracy'
    classifiers = []

    # 尝试使用SVC
    svc = SVC(random_state=random_state, probability=True)
    # 确定SVC参数列表
    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}
    best_cls_svc = GridSearchCV(svc, parameters, scoring=scorer).fit(x_train, y).best_estimator_
    print(best_cls_svc)
    classifiers.append(best_cls_svc)

    # 尝试使用RandomForestClassifier
    rf = RandomForestClassifier(random_state=random_state)
    # 确定rf参数列表
    parameters = {'max_depth': [2, 5, 10], 'n_estimators': [10, 100, 200]}
    best_cls_rf = GridSearchCV(rf, parameters, scoring=scorer).fit(x_train, y).best_estimator_
    print(best_cls_rf)
    classifiers.append(best_cls_rf)

    # 尝试使用MLPClassifier
    mlp = MLPClassifier(random_state=random_state)
    # 确定mlp参数列表
    parameters = {'activation': ('identity', 'logistic', 'tanh', 'relu'), 'hidden_layer_sizes': [10, 100, 200]}
    best_cls_mlp = GridSearchCV(mlp, parameters, scoring=scorer).fit(x_train, y).best_estimator_
    print(best_cls_mlp)
    classifiers.append(best_cls_mlp)

    # 尝试使用KNeighborsClassifier
    knn = KNeighborsClassifier()
    # 确定knn参数列表
    parameters = {'n_neighbors': [3, 5, 10]}
    best_cls_knn = GridSearchCV(knn, parameters, scoring=scorer).fit(x_train, y).best_estimator_
    print(best_cls_knn)
    classifiers.append(best_cls_knn)

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, x_train, y=y, scoring="accuracy", cv=kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC",
                                                                            "RandomForest",
                                                                            "MultipleLayerPerceptron", "KNeighboors"]})

    # g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
    # g.set_xlabel("Mean Accuracy")
    # g = g.set_title("Cross validation scores")
    print(cv_res)
    base_model = VotingClassifier(estimators=[('svc', best_cls_svc), ('rf', best_cls_rf),
                                              ('mlp', best_cls_mlp), ('knn', best_cls_knn)],
                                  voting='soft', weights=[2, 1, 1, 2])
    base_model.fit(x_train, y)
    res = cross_val_score(base_model, x_train, y=y, scoring="accuracy", cv=kfold, n_jobs=4)
    print('baseline show')
    print(res)
    pickle.dump(base_model, open('base_model', 'wb'))


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(num_folds, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = all_data[all_data['TARGET'].notnull()]
    test_df = all_data[all_data['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier()

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug=False):
    num_rows = 10000 if debug else None
    try:
        all_data = load(open('clear.dat', 'rb'))
    except Exception as err:
        preprocess_data()
    finally:
        all_data = feature_engineering()
        train_df = all_data[all_data['TARGET'].notnull()]
        test_df = all_data[all_data['TARGET'].isnull()]
        useful_cols=num_cols+cate_cols
        x_train=train_df[useful_cols]
        y=train_df[target_col]
        with timer("modelling"):
            modelling_classification(x_train, y)
        # with timer("modelling"):
        #     feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)


if __name__ == "__main__":
    # submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main(debug=True)