import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import TargetEncoder

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

params_hgb = {'l2_regularization': 1.5,
              'learning_rate': 0.1,
              'max_depth': 25,
              'max_iter': 1500}
params_rf = {'max_depth': 25,
             'max_features': 'sqrt',
             'min_samples_leaf': 3,
             'min_samples_split': 8, 
             'n_estimators': 45}

hgb = HistGradientBoostingClassifier(**params_hgb)
rf = RandomForestClassifier(**params_rf)

hgb.fit(X_train, y_train)
rf.fit(X_train, y_train)

tg_encoder = TargetEncoder()

X_train_stacked = pd.DataFrame({"hgb": tg_encoder.fit_transform(hgb.predict(X_train).reshape(-1, 1), y_train), 
                                "rf": tg_encoder.transform(rf.predict(X_train).reshape(-1, 1), y_train)})
y_train_stacked = train["Event"]

X_test_stacked = pd.DataFrame({"hgb": tg_encoder.transform(hgb.predict(X_test).reshape(-1, 1), y_test), 
                               "rf": tg_encoder.transform(rf.predict(X_test).reshape(-1, 1), y_test)})
y_test_stacked = test["Event"]

params = {
    'penalty': ['l2'],
    'C': [i for i in np.linspace(205, 215, 30)], #best is 209.48275862068965
    'solver': ['newton-cg'],
    'warm_start': [True],
    'max_iter': [10000]
}

gs = GridSearchCV(LogisticRegression(),
                  param_grid=params,
                  cv=StratifiedKFold(n_splits=2),
                  scoring="balanced_accuracy",
                  verbose=1)

gs.fit(X_train_stacked, y_train_stacked)
print("Score: ", gs.score(X_test_stacked, y_test_stacked), "\n\n")
print(gs.best_params_)

dump(gs.best_estimator_, savedir + "logreg_stacked_best_0.joblib")
dump(gs.best_params_, savedir + "logreg_stacked_params_0.joblib")
          