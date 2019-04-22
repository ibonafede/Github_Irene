#!/usr/bin/evn python

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Genera punti 3D. Attenzione le prime due coordinate sono usate come feature vector (X), la terza come valore da predirre (y). 
# Pertanto siamo nel caso d = 2
#   
mean = np.array([0.0,0.0,0.0])
cov = np.array([[1.2,-0.5,0.8], [-0.5,1.3,0.0], [0.8,0.0,1.0]])
data = np.random.multivariate_normal(mean, cov, 50)

# regular grid covering the domain of the data (usato per disegnare il risultato come una superfice)
X1,X2 = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
X1f = X1.flatten()
X2f = X2.flatten()

order = 2    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    X = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]     # matrice nel formato di pagina 5 delle dispense, data[:,0] e data[:,1] sono le prime due colonne, relative alla due feature
    y = data[:,2]
    Beta,_,_,_ = scipy.linalg.lstsq(X, y)               # risolve il problema ai minimi quadrati (pagina 6 delle dispense) e ritorna i coefficienti Beta. 
    
    # evaluate it on grid. Predizione del valore di tutti i nodi della mesh.
    Z = Beta[0]*X1 + Beta[1]*X2 + Beta[2]
    
    # or expressed using matrix/vector product
    # Z = np.dot(np.c_[X1f, X2f, np.ones(X1f.shape)], Beta).reshape(X1.shape)

elif order == 2:
    # best-fit quadratic curve
    X = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    Beta,_,_,_ = scipy.linalg.lstsq(X, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(X1f.shape), X1f, X2f, X1f*X2f, X1f**2, X2f**2], Beta).reshape(X1.shape)

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()
