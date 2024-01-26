import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from joblib import dump

basedir = r"data/"
savedir = r"params/"

train = pd.read_csv(basedir + 'train_preprocessed.csv')
test = pd.read_csv(basedir + 'test_preprocessed.csv')
groups = pd.read_csv(basedir + 'train.csv')["SampleID"].astype("category").cat.codes.to_numpy()

X_train = train.drop(["Event"], axis=1)
y_train = train["Event"]

X_test = test.drop(["Event"], axis=1)
y_test = test["Event"]

majors = ['SiO2_normalized', 'TiO2_normalized', 'Al2O3_normalized',
          'FeOT_normalized',
          # 'FeO_normalized', 'Fe2O3_normalized', 'Fe2O3T_normalized',
          'MnO_normalized', 'MgO_normalized', 'CaO_normalized',
          'Na2O_normalized', 'K2O_normalized',
          # 'P2O5_normalized','Cl_normalized'
          ]
traces = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'La',
          'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb',
          'Th', 'U']

majors_idx = [X_train.columns.get_loc(c) for c in majors if c in X_train]
traces_idx = [X_train.columns.get_loc(c) for c in traces if c in X_train]

params = {
    'n_estimators': [ 40, 45, 50],
    'max_depth': [25],
    'min_samples_split': [8],
    'min_samples_leaf': [3],
    'max_features': ['log2', 'sqrt']
}

gs = GridSearchCV(RandomForestClassifier(),
                  param_grid=params,
                  cv=StratifiedGroupKFold(n_splits=2),
                  scoring="balanced_accuracy",
                  verbose=1)

gs.fit(X_train, y_train, groups=groups)
print("Score: ", gs.score(X_test, y_test), "\n\n")
print(gs.best_params_)

dump(gs.best_estimator_, savedir + "svc_best_0.joblib")
dump(gs.best_params_, savedir + "svc_params_0.joblib")
          