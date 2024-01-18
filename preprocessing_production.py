import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

basedir = r"data/"
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

def get_major_traces(data, 
                     majors=majors, 
                     traces=traces):
    data_majors = data.loc[:, majors]
    data_traces = data.loc[:, traces]
    label = data.loc[:, 'Event']
    new_data = pd.concat([data_majors, data_traces, label], axis=1)
    return new_data

train = pd.read_csv(basedir + "train.csv")
test = pd.read_csv(basedir + "test.csv")
majors_idx = [train.columns.get_loc(c) for c in majors if c in train]
traces_idx = [train.columns.get_loc(c) for c in traces if c in train]

train = get_major_traces(train)
test = get_major_traces(test)

iter_impt = IterativeImputer(random_state=0, estimator=BayesianRidge(), max_iter=10)
smpl_impt = SimpleImputer(strategy="median")

train[majors] = iter_impt.fit_transform(train[majors])
test[majors] = iter_impt.transform(test[majors])
train[traces] = smpl_impt.fit_transform(train[traces])
test[traces] = smpl_impt.transform(test[traces])

X_train = train.loc[:, train.columns != 'Event']
y_train = train["Event"]
X_test = test.loc[:, test.columns != 'Event']
y_test = test["Event"]

feature_selector = SelectFromModel(RandomForestClassifier(random_state=56))

X_train  = feature_selector.fit_transform(X_train, y_train)
X_test  = feature_selector.transform(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

train = pd.concat((X_train, y_train), axis=1)
test = pd.concat((X_test, y_test), axis=1)

train.to_csv(basedir + "train_preprocessed.csv", index=False)
test.to_csv(basedir + "test_preprocessed.csv", index=False)