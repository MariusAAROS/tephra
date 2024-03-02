from flaml import AutoML
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from joblib import dump

basedir = r"data/"
savedir = r"params/"

train = pd.read_csv(basedir + 'stacked_train.csv')
test = pd.read_csv(basedir + 'stacked_test.csv')

X_train = train.drop(["3"], axis=1)
y_train = train["3"]

X_test = test.drop(["3"], axis=1)
y_test = test["3"]

params = {"C": np.logspace(-3, 3, 10), 
          "penalty": ["l1", "l2", "elasticnet"], 
          "max_iter": [10000]}

gs = GridSearchCV(LogisticRegression(),
                  param_grid=params,
                  cv=3)

gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)