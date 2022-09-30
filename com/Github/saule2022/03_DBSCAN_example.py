# pip install numpy
# pip install opencv-python
# pip install matplotlib
# pip install sklearn

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pathlib

from sklearn.cluster import DBSCAN, KMeans

image_path = pathlib.Path(r'c:\Users\ArturasKatvickis\PycharmProjects\HyaDS\intro\sandbox\pt_dpsi_2009.jpg')

orig_image = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures = 5000)

kp, des = orb.detectAndCompute(orig_image, None)

img_kp = cv2.drawKeypoints(orig_image, kp, None, color=(0,0,255), flags=0)
cv2.imwrite('temp_kp.jpg', img_kp)

points = np.int0([np.array(p.pt) for p in kp])
(w, h) = orig_image.shape[0:2]
rel_points = [[p[0]/h, p[1]/w] for p in points]

cluster = DBSCAN(eps=0.025,
                 min_samples=10,
                 n_jobs=-1)

labels = cluster.fit_predict(rel_points)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: {}'.format(n_clusters_))
print('Estimated number of noise points: {}'.format(n_noise_))

cluster_dict = {i:[] for i in range(n_clusters_)}
for nr, label in enumerate(labels):
    if label != -1:
        cluster_dict[label].append(points[nr])

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for nr, label in enumerate(labels):
    if label == -1:
        col = [0, 0, 0, 1]
    else:
        col = colors[label]
    plt.plot(points[nr][0], w-points[nr][1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.axis('equal')
plt.savefig('clusters.jpg')
plt.close()

print('DONE!!!')