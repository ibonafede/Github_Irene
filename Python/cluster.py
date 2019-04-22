import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist,squareform

km=KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(expression)
distorsion=[]
for i in range(1,11):
	km=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
	km.fit(tmp)
	distorsion.append(km.inertia_)

plt.plot(range(1,11),distorsion,marker='o')
plt.show()

plt.legend()
plt.grid()
plt.show()

row_dist=pd.DataFrame(squareform(pdist(tmp,metric='euclidean')),columns=tmp.index,index=tmp.index)

row_clusters=linkage(row_dist,method='complete',metric='euclidean')
r_dendr=dendrogram(row_clusters,labels=tmp.index)
plt.tight_layout()
plt.show()

