import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Questo script utilizza la classe LinearRegression di scikit learn che espone la stessa interfaccia dei classificatori
#  con metodi fit e predict 
regr = linear_model.LinearRegression()

# In questo caso l'array X non deve essere creato con la colonna di 1 finale (come nelle dispense), ci pensa la libreria a farlo internamente.
X = [[72], [54], [65], [58], [62], [72], [60], [64], [55], [46], [52], [47], [55], [54], [55], [49], [51], [51]]
y = [173, 159, 172, 170, 165, 176, 173, 179, 166, 158, 158, 160, 167, 166, 164, 157, 165, 162]

# Train the model using the training sets
regr.fit (X,y)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f" % np.mean((regr.predict(X) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X, y))

# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X, regr.predict(X), color='blue', linewidth=1)
plt.title('Predizione Altezza da Peso')
plt.xlabel('Peso')
plt.ylabel('Altezza')

plt.xticks(())
plt.yticks(())
plt.show()

