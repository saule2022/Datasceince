import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt

data_path = '/Users/daivadaugelaite/Downloads/wateredit.xlsx'
df = pd.read_excel(data_path)
#'ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity','Potability'
pd.plotting.scatter_matrix(df[['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity','Potability']])

cor_matr = df.corr()

names2 = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity','Potability']
df2 = df[names2].copy()


#data_path = 'C:/DATA/Hyarchis_DS_Academy/water_potability.csv'

# nuskaitome duomenys
#df = pd.read_csv(data_path)

# pasitikriname ar yra (ir kiek yra) missing values
df.info()

# pasaliname eilutes kur yra missing values
df.dropna(inplace=True)
# df['Potability']

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
plt.show()

# apmokome Random Forest modeli
RF = RandomForestClassifier(n_estimators=50,
                            min_samples_split=5,
                            min_samples_leaf=5,
                            n_jobs=-1)
RF = RF.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
RF_prediction = RF.predict(X_test_transformed)
print(confusion_matrix(RF_prediction, Y_test))
print(classification_report(RF_prediction, Y_test))

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

# apmokame keliu algoritmu misini
clf1 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf2 = SVC(kernel='linear')
clf3 = SVC(kernel='rbf')
clf4 = SVC(kernel='poly', degree=2)
clf5 = GaussianNB()
voting = VotingClassifier(estimators=[('rf', clf1),
                                      ('svc_linear', clf2),
                                      ('svc_rbf', clf3),
                                      ('svc_poly', clf4),
                                      ('nb', clf5)],
                          voting='soft',
                          n_jobs=-1)
voting = voting.fit(X_train_transformed, Y_train)

# apskaiciuojame modelio charakteristikas
naiveBayes_prediction = naiveBayes.predict(X_test_transformed)
print(confusion_matrix(naiveBayes_prediction, Y_test))
print(classification_report(naiveBayes_prediction, Y_test))

RocCurveDisplay.from_estimator(naiveBayes, X_test_transformed, Y_test)
plt.grid()
plt.axis('equal')
plt.show()

# nusistatyti optimalu threshold
SVC_poly_probabilities = SVC_poly.predict_proba(X_test_transformed)
prob = pd.DataFrame(SVC_poly_probabilities, columns=['0', '1'])
fpr, tpr, thresholds = roc_curve(Y_test, prob['1'])

