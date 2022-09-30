import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data_path = 'C:\DATA\Hyarchis_DS_Academy\Raisin_Dataset\Raisin_Dataset.csv'
df = pd.read_csv(data_path)

pd.plotting.scatter_matrix(df[['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']])
df['target'] = df['Class'].apply(lambda x : (x == 'Kecimen')*1)

X_train, X_test, Y_train, Y_test = train_test_split(df[['Area',
                                                        'MajorAxisLength',
                                                        'MinorAxisLength',
                                                        'Eccentricity',
                                                        'ConvexArea',
                                                        'Extent',
                                                        'Perimeter']],
                                                    df['target'], test_size=0.2, random_state=42)

model = Pipeline([('scaler', StandardScaler()),
                  ('clf', RandomForestClassifier(n_estimators=50,
                                                 min_samples_split=5,
                                                 min_samples_leaf=5,
                                                 n_jobs=-1))])

model.fit(X_train, Y_train)

prob = model.predict_proba(X_test)
df_prob = pd.DataFrame(prob, columns=['0', '1'])
Y_test_lst = list(Y_test)
df_prob['actual'] = Y_test_lst
df_prob['pred_50'] = df_prob['1'].apply(lambda x: (x > 0.5)*1)

confusion_matrix(df_prob['actual'], df_prob['pred_50'])
print(classification_report(df_prob['actual'], df_prob['pred_50']))

df_prob['pred_60'] = df_prob['1'].apply(lambda x: (x > 0.6)*1)
print(classification_report(df_prob['actual'], df_prob['pred_60']))
print(confusion_matrix(df_prob['actual'], df_prob['pred_60']))

[73, 13],
[12, 82]]

[[76 10]
 [15 79]]