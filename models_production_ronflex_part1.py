from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from joblib import load, dump

basedir = r"data/"
savedir = r"params/"
training = False

settings_xgb = {'objective': 'multi:softprob', 
                'base_score': None, 
                'booster': None, 
                'callbacks': [], 
                'colsample_bylevel': 0.3595514135915155, 
                'colsample_bynode': None, 
                'colsample_bytree': 0.7048987333256599, 
                'device': None, 
                'early_stopping_rounds': None, 
                'enable_categorical': True, 
                'eval_metric': None, 
                'feature_types': None, 
                'gamma': None, 
                'grow_policy': 
                'lossguide', 
                'importance_type': None, 
                'interaction_constraints': None, 
                'learning_rate': 0.11947186047935418, 
                'max_bin': None, 
                'max_cat_threshold': None, 
                'max_cat_to_onehot': None, 
                'max_delta_step': None, 
                'max_depth': 0, 
                'max_leaves': 12, 
                'min_child_weight': 0.0038997862899286576, 
                'monotone_constraints': None, 
                'multi_strategy': None, 
                'n_estimators': 2784, 
                'n_jobs': -1, 
                'num_parallel_tree': None, 
                'random_state': None, 
                'reg_alpha': 0.002359632411866371, 
                'reg_lambda': 1.225037482688091, 
                'sampling_method': None, 
                'scale_pos_weight': None, 
                'subsample': 0.324662607343124, 
                'tree_method': 'hist', 
                'validate_parameters': None, 
                'verbosity': 0}

settings_cb = {'learning_rate': 0.009995958806260535, 
               'random_seed': 10242048, 
               'verbose': False, 
               'train_dir': 'catboost_1709316880.174755', 
               'n_estimators': 5023, 
               'early_stopping_rounds': 39}

settings_lgbm = {'boosting_type': 'gbdt', 
                 'class_weight': None, 
                 'colsample_bytree': 0.773947204068062, 
                 'importance_type': 'split', 
                 'learning_rate': 0.027580294624245914, 
                 'max_depth': -1, 
                 'min_child_samples': 6, 
                 'min_child_weight': 0.001, 
                 'min_split_gain': 0.0, 
                 'n_estimators': 1, 
                 'n_jobs': -1, 
                 'num_leaves': 17, 
                 'objective': None, 
                 'random_state': None, 
                 'reg_alpha': 0.009351976587500649, 
                 'reg_lambda': 0.009557995081783356, 
                 'subsample': 1.0, 
                 'subsample_for_bin': 200000, 
                 'subsample_freq': 0, 
                 'max_bin': 127, 
                 'verbose': -1}

le = LabelEncoder()
train = pd.read_csv(basedir + 'train_preprocessed.csv')
test = pd.read_csv(basedir + 'test_preprocessed.csv')
groups = pd.read_csv(basedir + 'train.csv')["SampleID"].astype("category").cat.codes.to_numpy()

X_train = train.drop(["Event"], axis=1)
y_train = train["Event"]
y_train_cat = le.fit_transform(y_train)

X_test = test.drop(["Event"], axis=1)
y_test = test["Event"]
y_test_cat = le.transform(y_test)

xgb = XGBClassifier(**settings_xgb)
cb = CatBoostClassifier(**settings_cb)
lgbm = LGBMClassifier(**settings_lgbm)

if training:
    xgb.fit(X_train.values, y_train_cat)
    cb.fit(X_train.values, y_train_cat)
    lgbm.fit(X_train.values, y_train_cat)
    dump(xgb, savedir + "xgb.joblib")
    dump(cb, savedir + "cb.joblib")
    dump(lgbm, savedir + "lgbm.joblib")
else:
    xgb = load(savedir + "xgb.joblib")
    cb = load(savedir + "cb.joblib")
    lgbm = load(savedir + "lgbm.joblib")

xgb_pred = xgb_pred = xgb.predict(X_test)
cb_pred = cb.predict(X_test)
lgbm_pred = lgbm.predict(X_test)

stacked_X_train = np.column_stack((xgb.predict(X_train.values), 
                                   cb.predict(X_train.values), 
                                   lgbm.predict(X_train.values),
                                   y_train))
stacked_X_test = np.column_stack((xgb_pred, 
                                  cb_pred, 
                                  lgbm_pred, 
                                  y_test))

pd.DataFrame(stacked_X_train).to_csv(basedir + "stacked_train.csv", index=False)
pd.DataFrame(stacked_X_test).to_csv(basedir + "stacked_test.csv", index=False)