import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data_path = pathlib.Path(r'C:\repos\HyaDS\intro\sandbox\players2013.csv')

df = pd.read_csv(data_path.as_posix())


df.info()

stdScale = StandardScaler()
df1 = pd.DataFrame(stdScale.fit_transform(df[['FSP', 'ACE']]),
                   columns=['FSP', 'ACE'])

data = df1[['FSP', 'ACE']].copy()
data.dropna(inplace=True)

distances = []
silhouette = []
K = range(2,20)
for nr in K:
    kmeans = KMeans(n_clusters=nr)
    kmeans.fit(data)
    distances.append(kmeans.inertia_)
    silhouette.append(silhouette_score(data, kmeans.labels_))

fig,ax = plt.subplots()
ax.plot(K,distances, 'ro--')
ax2=ax.twinx()
ax2.plot(K,silhouette,'bo--')
plt.show()

N_clusters = 3
kmeans = KMeans(n_clusters=N_clusters)
kmeans.fit(data)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, N_clusters)]
FSP_values = list(data['FSP'])
ACE_values = list(data['ACE'])
for nr, label in enumerate(kmeans.labels_):
    col = colors[label]
    plt.plot(FSP_values[nr], ACE_values[nr], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
plt.show()