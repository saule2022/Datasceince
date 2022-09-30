import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_curve


data_path = 'C:/DATA/Hyarchis_DS_Academy/water_potability.csv'

# nuskaitome duomenys
df = pd.read_csv(data_path)

# pasitikriname ar yra (ir kiek yra) missing values
df.info()

# pasaliname eilutes kur yra missing values
df.dropna(inplace=True)
df['Potability'].plot.hist()

# isskirtome i train ir test dalys
Y = df['Potability']
X = df.drop(columns=['Potability'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# nuscale'iname duomenys - 'apmokome' StandartScaler'i ir transformuojame tiek train tiek test rinkinius
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# apmokome Logistines regresijos modeli
LR = LogisticRegression(n_jobs=-1)
LR = LR.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
LR_prediction = LR.predict(X_test_transformed)
print(confusion_matrix(LR_prediction, Y_test))
print(classification_report(LR_prediction, Y_test))

# Metrikos yra blogos, default threshold vale = 0.5 reiksme gali b8ti netinkama
# Tam kad ivertinti optimalia reiksme braizome ROC kreive
RocCurveDisplay.from_estimator(LR, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()

# apmokome Random Forest modeli
RF = RandomForestClassifier(n_estimators=50,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            n_jobs=-1)
RF = RF.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
RF_prediction = RF.predict(X_test_transformed)
print(confusion_matrix(RF_prediction, Y_test))
print(classification_report(RF_prediction, Y_test))
# [[188  98]
#  [ 43  74]]
#               precision    recall  f1-score   support
#            0       0.81      0.66      0.73       286
#            1       0.43      0.63      0.51       117
#     accuracy                           0.65       403
#    macro avg       0.62      0.64      0.62       403
# weighted avg       0.70      0.65      0.66       403

RocCurveDisplay.from_estimator(RF, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()

# apmokome linear SVC modeli
SVC_linear = SVC(kernel='linear', probability=True)
SVC_linear = SVC_linear.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
SVC_prediction = SVC_linear.predict(X_test_transformed)
print(confusion_matrix(SVC_prediction, Y_test))
print(classification_report(SVC_prediction, Y_test))

RocCurveDisplay.from_estimator(SVC_linear, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()

# apmokome SVC modeli su rbf branduolio funkcija
SVC_rbf = SVC(kernel='rbf', probability=True)
SVC_rbf = SVC_rbf.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
SVC_rbf_prediction = SVC_rbf.predict(X_test_transformed)
print(confusion_matrix(SVC_rbf_prediction, Y_test))
print(classification_report(SVC_rbf_prediction, Y_test))

RocCurveDisplay.from_estimator(SVC_rbf, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()
# [[207 108]
#  [ 24  64]]
#               precision    recall  f1-score   support
#            0       0.90      0.66      0.76       315
#            1       0.37      0.73      0.49        88
#     accuracy                           0.67       403
#    macro avg       0.63      0.69      0.63       403
# weighted avg       0.78      0.67      0.70       403

# apmokome SVC modeli su poly branduolio funkcija
SVC_poly = SVC(kernel='poly', degree=2, probability=True)
SVC_poly = SVC_poly.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
SVC_poly_prediction = SVC_poly.predict(X_test_transformed)
print(confusion_matrix(SVC_poly_prediction, Y_test))
print(classification_report(SVC_poly_prediction, Y_test))

RocCurveDisplay.from_estimator(SVC_poly, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()
# [[219 112]
#  [ 12  60]]
#               precision    recall  f1-score   support
#            0       0.95      0.66      0.78       331
#            1       0.35      0.83      0.49        72
#     accuracy                           0.69       403
#    macro avg       0.65      0.75      0.64       403
# weighted avg       0.84      0.69      0.73       403

# apmokome GaussianNB modeli
naiveBayes = GaussianNB()
naiveBayes = naiveBayes.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
naiveBayes_prediction = naiveBayes.predict(X_test_transformed)
print(confusion_matrix(naiveBayes_prediction, Y_test))
print(classification_report(naiveBayes_prediction, Y_test))

RocCurveDisplay.from_estimator(naiveBayes, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()
# [[209 130]
#  [ 22  42]]
#               precision    recall  f1-score   support
#            0       0.90      0.62      0.73       339
#            1       0.24      0.66      0.36        64
#     accuracy                           0.62       403
#    macro avg       0.57      0.64      0.54       403
# weighted avg       0.80      0.62      0.67       403


# apmokame keliu algoritmu misini
clf1 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
# clf2 = SVC(kernel='linear')
clf3 = SVC(kernel='rbf', probability=True)
clf4 = SVC(kernel='poly', degree=2, probability=True)
# clf5 = GaussianNB()
voting = VotingClassifier(estimators=[('rf', clf1),
                                      ('svc_rbf', clf3),
                                      ('svc_poly', clf4)],
                          voting='soft',
                          n_jobs=-1)
voting = voting.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
voting_prediction = voting.predict(X_test_transformed)
print(confusion_matrix(voting_prediction, Y_test))
print(classification_report(voting_prediction, Y_test))

RocCurveDisplay.from_estimator(voting, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()
# [[207  96]
#  [ 24  76]]
#               precision    recall  f1-score   support
#            0       0.90      0.68      0.78       303
#            1       0.44      0.76      0.56       100
#     accuracy                           0.70       403
#    macro avg       0.67      0.72      0.67       403
# weighted avg       0.78      0.70      0.72       403



# nusistatyti optimalu threshold
SVC_poly_probabilities = SVC_poly.predict_proba(X_test_transformed)
prob = pd.DataFrame(SVC_poly_probabilities, columns=['0', '1'])
fpr, tpr, thresholds = roc_curve(Y_test, prob['1'])
# fpr ~ 0.6; tpr ~ 0.22

prob['predict_0.35'] = prob['1'].apply(lambda x: (x > 0.35)*1)

print(confusion_matrix(prob['predict_0.35'], Y_test))
print(classification_report(prob['predict_0.35'], Y_test))
# [[181  70]
#  [ 50 102]]
#               precision    recall  f1-score   support
#            0       0.78      0.72      0.75       251
#            1       0.59      0.67      0.63       152
#     accuracy                           0.70       403
#    macro avg       0.69      0.70      0.69       403
# weighted avg       0.71      0.70      0.71       403