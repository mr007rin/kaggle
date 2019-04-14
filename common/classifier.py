import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import VotingClassifier, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def kfold_sklearn_clf(x_train, y):
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
def kfold_lightgbm_clf(df, num_folds, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
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

