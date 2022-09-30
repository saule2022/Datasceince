import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from scipy import interpolate

# https://archive.ics.uci.edu/ml/datasets/Glass+Identification
data_path = pathlib.Path(r'C:\DATA\Hyarchis_DS_Academy\glass.data')

col_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']

data = pd.read_csv(data_path.as_posix(), header=None, names=col_names)
# pd.plotting.scatter_matrix(data)

label_enc = LabelEncoder()
target = label_enc.fit_transform(data['class'])
n_classes = len(label_enc.classes_)
features = data[col_names[1:-1]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.50, random_state=42)

clf = RandomForestClassifier(n_jobs=-1)
RF = clf.fit(X_train, y_train)

y_prob = RF.predict_proba(X_test)
y_pred = RF.predict(X_test)

y_predictions = label_enc.inverse_transform(y_pred)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(multilabel_confusion_matrix(y_test, y_pred))

print(roc_auc_score(y_test, y_prob, multi_class="ovo"))
print(roc_auc_score(y_test, y_prob, multi_class="ovr"))

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_classes)]
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,i], pos_label=i)
    print(thresholds)
    print('\n\n')
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], label=f'Class {str(i)} , AUC={str(area)}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.axis('equal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()




#########################################################################
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_test, y_prob[:,i], pos_label=i)

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    f = interpolate.interp1d(fpr[i], tpr[i])
    mean_tpr += [f(x) for x in all_fpr]

mean_tpr /= n_classes

plt.plot(all_fpr, mean_tpr, 'k--')
plt.plot([0, 1], [0, 1], 'k*')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.axis('equal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print(auc(all_fpr, mean_tpr))