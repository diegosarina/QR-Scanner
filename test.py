import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import distance
import pdb
a = np.matrix([[1,1],[10,10],[20,20]])
b = np.matrix([10,10])
distancias = [np.linalg.norm(j-b) for j in a]
y = distancias.index(min(distancias))

z = pdist(a)

i = np.argmax(z)
print(z[1])
print(i)
print(z)