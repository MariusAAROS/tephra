import pandas as pd

from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer

majors = [i for i in range(9)]
traces = [i for i in range(9, 35)]
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

class Classifier(BaseEstimator):
    def __init__(self):
        self.transformer = Pipeline([
            ("impute_scale", ColumnTransformer([
                ("imputer_majors", IterativeImputer(random_state=0, estimator=BayesianRidge(), max_iter=10), majors),
                ("imputer_traces", SimpleImputer(strategy="median"), traces),
            ], remainder='passthrough')),
            ("feature_selection", SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=10000))),
            ("scaler", StandardScaler()),
        ])
        self.estimators = [
            ("xgboost", XGBClassifier(**settings_xgb)),
            ("catboost", CatBoostClassifier(**settings_cb)),
            ("lightgbm", LGBMClassifier(**settings_lgbm))
        ]
        self.model = VotingClassifier(estimators=self.estimators, voting="soft")
        self.pipe = make_pipeline(self.transformer, self.model)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        X = X.drop(["groups"], axis=1)
        y = self.label_encoder.fit_transform(y)
        self.pipe.fit(X, y)

    def predict(self, X):
        X = X.drop(["groups"], axis=1)
        pred = self.pipe.predict(X)
        return self.label_encoder.inverse_transform(pred)

    def predict_proba(self, X):
        X = X.drop(["groups"], axis=1)
        return self.pipe.predict_proba(X)