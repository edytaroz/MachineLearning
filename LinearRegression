from math import sqrt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


regSci = LinearRegression(fit_intercept=True)

one = [1,1,1,1,1,1,1,1,1,1]
f1 = [2.31,7.07,7.07,2.18,2.18,2.18,7.87,7.87,7.87,7.87]
f2 = [65.2,78.9,61.1,45.8,54.2,58.7,96.1,100.0,85.9,94.3]
f3 = [15.3,17.8,17.8,18.7,18.7,18.7,15.2,15.2,15.2,15.2]
y = [24.0,21.6,34.7,33.4,36.2,28.7,27.1,16.5,18.9,15.0]
d = pd.DataFrame({'f1':np.array(f1),'f2':np.array(f2),'f3':np.array(f3),'y':np.array(y)})


class CustomLinearRegression:

    def __init__(self, *, fit_intercept):

        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = None

    def fit(self, X, y):
        xt = np.transpose(X)
        B = np.matmul(np.asarray(xt),X)
        B1 = np.linalg.inv(B)
        B2 = np.matmul(B1,xt)
        Bfin = np.matmul(B2,y)
        self.coefficient = np.array(Bfin)
        if self.fit_intercept:
            self.intercept = Bfin[0]

    def predict(self, X):
        pred = np.matmul(X,self.coefficient)
        return np.array(pred)

    def r2_score(self, y, yhat):
        m = np.mean(y)
        suma = 0
        suma1 = 0
        n = len(y)
        for i in range(n):
            suma += (y[i] - yhat[i])*(y[i] - yhat[i])
            suma1 += (y[i] - m)*(y[i] - m)
        return 1 - suma/suma1

    def rmse(self, y, yhat):
        n = len(y)
        suma = 0
        for i in range(n):
            suma += (y[i] - yhat[i])*(y[i] - yhat[i])
        suma = suma / n
        return sqrt(suma)

lin = CustomLinearRegression(fit_intercept=True)

if lin.fit_intercept:
    Xl = pd.DataFrame({'one':np.array(one),'f1':d['f1'],'f2':d['f2'],'f3':d['f3']})
    
else:
   X1 = pd.DataFrame({'f1':d['f1'],'f2':d['f2'],'f3':d['f3']})



X = pd.DataFrame({'f1':d['f1'],'f2':d['f2'],'f3':d['f3']})
regSci.fit(X,d['y'])
lin.fit(Xl,d['y'])
y_pred = lin.predict(Xl)
ypred = regSci.predict(X)
r2sc = r2_score(d['y'],y_pred)
rmse1 = mean_squared_error(d['y'],y_pred)
r2sc2 = r2_score(d['y'],ypred)
rmse2 = mean_squared_error(d['y'],ypred)
