# Classification

# 1.Choose a relatively high learning rate
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve

param = {
    'learning_rate': 0.01,
    'objective': 'binary',
    'metric': 'auc',
}
xg_train = lgb.Dataset(X_train.values,
                       label=y_train.values,
                       )
xg_valid = lgb.Dataset(X_test.values,
                       y_test.values,
                       )

num_round = 1000

clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval=100, early_stopping_rounds = 500)

print('best_iteration: '+str(clf.best_iteration))
print('best_score: '+str(clf.best_score))

# 2.use BayesianOptimization to get best treebased paras
from bayes_opt import BayesianOptimization
def bayes_parameter_opt_lgb(init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000,
                            learning_rate=0.05, output_process=False):
    # prepare data
    train_data = xg_train

    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain,
                 min_child_weight):
        params = {'application': 'binary', 'num_boost_round': n_estimators, 'learning_rate': learning_rate,
                  'early_stopping_round': 100, 'metric': 'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval=200,
                           metrics=['auc'])
        return max(cv_result['auc-mean'])

    # range
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    # return best parameters
    return lgbBO


opt_params = bayes_parameter_opt_lgb(init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=22,
                                     learning_rate=0.05)
print(opt_params.max)


# Regression
import numpy as np
from sklearn.metrics import mean_squared_error

# define a val metrics func rmsle
def rmsle(preds, train_data):
    y=train_data.get_label()
    return 'error', np.sqrt(mean_squared_error(y, preds)), False

xg_train = lgb.Dataset(X_train.values,
                       label=y_train.values,
                       )
xg_valid = lgb.Dataset(X_val.values,
                       y_val.values,
                       )

from bayes_opt import BayesianOptimization

def bayes_parameter_opt_lgb(init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000,
                            learning_rate=0.05, output_process=False):

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_bin, bagging_freq):
        params = {'application': 'regression', 'num_boost_round': n_estimators, 'learning_rate': learning_rate,
                  'early_stopping_round': 100, 'metric': 'rmsle'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_bin'] = int(round(max_bin))

        clf = lgb.train(params, xg_train, n_estimators, valid_sets=[xg_valid], feval=rmsle, verbose_eval=100,
                        early_stopping_rounds=500)
        return 1 / clf.best_score['valid_0']['error']

    # range
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (4, 10),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.6, 0.9),
                                            'max_bin': (180, 220),
                                            'bagging_freq': (2, 7)}
                                 )
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    # return best parameters
    return lgbBO


opt_params = bayes_parameter_opt_lgb(init_round=5, opt_round=500, n_folds=3, random_seed=6, n_estimators=2694,
                                     learning_rate=0.01)
print(opt_params.max)
