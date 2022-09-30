import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from scipy import interpolate

# https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
train_path = pathlib.Path(r'C:\DATA\Hyarchis_DS_Academy\pendigits\pendigits.tra')
test_path = pathlib.Path(r'C:\DATA\Hyarchis_DS_Academy\pendigits\pendigits.tes')

col_names = [f'feature_{str(i).zfill(2)}' for i in range(16)]
col_names.append('target')
train_df = pd.read_csv(train_path.as_posix(), header=None, names=col_names)
test_df = pd.read_csv(test_path.as_posix(), header=None, names=col_names)

model = MLPClassifier(hidden_layer_sizes=(25, 5))

model.fit(train_df[col_names[:-1]], train_df['target'])

prob = model.predict_proba(test_df[col_names[:-1]])
pred = model.predict(test_df[col_names[:-1]])

print(confusion_matrix(test_df['target'], pred))
print(classification_report(test_df['target'], pred))