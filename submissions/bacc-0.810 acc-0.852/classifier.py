import pandas as pd

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer

majors = [i for i in range(9)]
traces = [i for i in range(9, 35)]
params_hgb = {'l2_regularization': 1.5,
              'learning_rate': 0.1,
              'max_depth': 25,
              'max_iter': 1500}
params_logreg = {'C': 209.48275862068965,
                 'max_iter': 10000,
                 'penalty': 'l2',
                 'solver': 'newton-cg',
                 'warm_start': True}

class Classifier(BaseEstimator):
    def __init__(self):
        self.transformer = Pipeline([
            ("impute_scale", ColumnTransformer([
                ("imputer_majors", IterativeImputer(random_state=0, estimator=BayesianRidge(), max_iter=10), majors),
                ("imputer_traces", SimpleImputer(strategy="median"), traces),
            ], remainder='passthrough')),
            ("feature_selection", SelectFromModel(LinearSVC(penalty="l1", dual=False))),
            ("scaler", StandardScaler()),
        ])
        self.estimators = [
            ("hist_gradient_boosting", HistGradientBoostingClassifier(**params_hgb)),
            ("logistic_regression", LogisticRegression(**params_logreg))
        ]
        self.model = StackingClassifier(estimators=self.estimators)
        self.pipe = make_pipeline(self.transformer, self.model)

    def fit(self, X, y):
        X = X.drop(["groups"], axis=1)
        self.pipe.fit(X, y)

    def predict(self, X):
        X = X.drop(["groups"], axis=1)
        return self.pipe.predict(X)

    def predict_proba(self, X):
        X = X.drop(["groups"], axis=1)
        return self.pipe.predict_proba(X)