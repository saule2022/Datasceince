import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_curve

train_path = pathlib.Path(r'C:\DATA\Hyarchis_DS_Academy\adult.data')
test_path = pathlib.Path(r'C:\DATA\Hyarchis_DS_Academy\adult.test')

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'income']

cat_names = ['workclass', 'education', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'native_country']

data = pd.read_csv(train_path.as_posix(), header=None, names=col_names)
pd.plotting.scatter_matrix(data)

data['native_country'].replace(to_replace=' ?', value='unknown', inplace=True)
data['workclass'].replace(to_replace=' ?', value='unknown', inplace=True)
data['occupation'].replace(to_replace=' ?', value='unknown', inplace=True)

data['workclass'].unique()

enc = OneHotEncoder()
trans = enc.fit_transform(data[cat_names])

RF = RandomForestClassifier()
model = RF.fit(trans, data['income'])

test = pd.read_csv(test_path.as_posix(), header=None, names=col_names)
test['native_country'].replace(to_replace=' ?', value='unknown', inplace=True)
test['workclass'].replace(to_replace=' ?', value='unknown', inplace=True)
test['occupation'].replace(to_replace=' ?', value='unknown', inplace=True)

test_trans = enc.transform(test[cat_names])
prediction = model.predict(test_trans)

confusion_matrix(test['income'], prediction)