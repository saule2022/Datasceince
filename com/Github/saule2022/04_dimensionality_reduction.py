import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline

data_path = pathlib.Path(r'C:\Users\ArturasKatvickis\PycharmProjects\HyaDS\intro\sandbox\players2013_withoutNaN.csv')

df = pd.read_csv(data_path.as_posix())

column_names = df.columns

stdScaler = StandardScaler()
df1 = pd.DataFrame(stdScaler.fit_transform(df[column_names[1:]]),columns=df.columns[1:])
df1['Player1'] = df['Player']

pca = PCA(n_components=9)
pca.fit(df[column_names[1:]])

print(pca.explained_variance_ratio_)
# [0.71054559 0.12380657 0.05646739 0.04566689 0.02713181 0.01367732
#  0.00646955 0.00399699 0.0030173 ]

sum(pca.explained_variance_ratio_)
# 0.9907794146815913

pca1 = PCA(n_components=9)
pca1.fit(df1[column_names[1:]])

print(pca1.explained_variance_ratio_)
# [0.40573173 0.14065232 0.0821764  0.06545694 0.05648088 0.04412659
#  0.0414759  0.0362907  0.03293294]


pca_5 = PCA(n_components=5)
pca_5.fit(df1[column_names[1:]])

print(pca_5.explained_variance_ratio_)
# [0.40573173 0.14065232 0.0821764  0.06545694 0.05648088]

print(sum(pca_5.explained_variance_ratio_))
# 0.7504982688090215

df_trans = pd.DataFrame(pca1.transform(df1[column_names[1:]]))

SVD = TruncatedSVD(n_components=9)
SVD.fit(df[column_names[1:]])

print(SVD.explained_variance_ratio_)
# [0.48078775 0.24179622 0.11553998 0.05645062 0.04222222 0.02713181
#  0.01342579 0.00643811 0.00399338]

sum(SVD.explained_variance_ratio_)
# 0.9877858712607498

SVD1 = TruncatedSVD(n_components=9)
SVD1.fit(df1[column_names[1:]])

print(SVD1.explained_variance_ratio_)
# [0.40573173 0.14065232 0.0821764  0.06545694 0.05648088 0.04412659
#  0.0414759  0.0362907  0.03293294]

sum(SVD1.explained_variance_ratio_)
# 0.9053244076268725

print(pca1.explained_variance_ratio_)
# [0.40573173 0.14065232 0.0821764  0.06545694 0.05648088 0.04412659
#  0.0414759  0.0362907  0.03293294]

sum(pca1.explained_variance_ratio_)
# 0.9053244076268726


joblib.dump(SVD1, 'svd.pkl')

model = joblib.load('svd.pkl')
df_model = pd.DataFrame(model.transform(df1[column_names[1:]]))