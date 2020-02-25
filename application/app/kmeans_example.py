from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pdb
import numpy as np
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }

Data2 = np.matrix([[25,79],[34,51],[22,53],[27,78],[33,59],[33,74],[31,73]])
a=np.matrix([[25,79],[34,51],[22,53],[27,78],[33,59],[33,74],[31,73]])

df = DataFrame(Data2,columns=['x','y'])

kmeans = KMeans(n_clusters=3).fit(df)
idx = kmeans.labels_
pdb.set_trace()
centroids = kmeans.cluster_centers_
#print(centroids)
print(idx)
print(idx == 1)
#plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.scatter(df['x'], df['y'],c=kmeans.labels_.astype(float),s=50, alpha=0.5)
plt.show()