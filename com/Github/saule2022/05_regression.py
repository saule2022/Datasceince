import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


names=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
X, Y = load_diabetes(return_X_y=True)
df = pd.DataFrame(data=X, columns=names)

pd.plotting.scatter_matrix(df)

cor_matr = df.corr()

names2 = ['age', 'bmi', 'bp', 's2', 's3', 's5', 's6']
df2 = df[names2].copy()

X_train, X_test, Y_train, Y_test = train_test_split(df2, Y, test_size=0.2, random_state=42)

LR = LinearRegression(n_jobs=-1)
LR_model = LR.fit(X_train, Y_train)

Y_pred = LR_model.predict(X_test)
R2_LR = r2_score(Y_test, Y_pred)
#0.4432913309976919

RF = RandomForestRegressor(n_jobs=-1)
RF_model = RF.fit(X_train, Y_train)

Y_pred2 = RF_model.predict(X_test)
R2_RF = r2_score(Y_test, Y_pred2)
#0.4281267018342655

df['target'] = Y
pd.plotting.scatter_matrix(df)
corr=df.corr()