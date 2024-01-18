import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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

params = {'max_iter': [15000],
          'learning_rate': [0.1],
          'max_depth' : [25, 30, 35],
          'l2_regularization': [2.0, 3.0, 4.0],
          'max_leaf_nodes': [30],
          'min_samples_leaf': [15],
          'max_bins': [220, 255, 290],
          'scoring': ['f1_micro']
}

gs = GridSearchCV(HistGradientBoostingClassifier(),
                  param_grid=params,
                  cv=StratifiedGroupKFold(n_splits=2),
                  scoring="balanced_accuracy",
                  verbose=1)

gs.fit(X_train, y_train, groups=groups)
print("Score: ", gs.score(X_test, y_test), "\n\n")
print(gs.best_params_)

dump(gs.best_estimator_, savedir + "HGB_best_1.joblib")
dump(gs.best_params_, savedir + "HGB_best_params_1.joblib")