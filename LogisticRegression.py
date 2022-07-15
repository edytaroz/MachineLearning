import pandas as pd
import numpy as np
from math import e
from math import log10
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

class CustomLogisticRegression:
    def __init__(self, fit_intercept, l_rate, n_epoch):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = np.array([0, 0, 0, 0])
        self.coef_1 = np.array([0, 0, 0, 0])
    def sigmoid(self,t):
        return 1/(1+e**(-t))
    def predict_proba(self,row,coef_):
        t = np.matmul(row,coef_[1:])
        if self.fit_intercept:
            t += coef_[0]
        return self.sigmoid(t)

    def fit_log_loss(self, X_train, y_train):
        bs = [0, 0, 0, 0]
        sumy = [[],[]]
        n = len(X_train)
        for k in range(self.n_epoch):
            suma = []
            for i, row in enumerate(X_train):
                coef_ = np.array(bs)
                dev = 0
                y_hat = self.predict_proba(row, coef_)
                val = y_train[i] * log10(y_hat) + (1 - y_train[i]) * log10(1 - y_hat)
                suma.append(val/(-n))
                for j in range(len(self.coef_)):
                    if j != 0:
                        dev = self.l_rate * (y_hat - y_train[i]) * row[j - 1] / n
                    else:
                        dev = self.l_rate * (y_hat - y_train[i]) / n
                    bs[j] -= dev
            if k == 0:
                sumy[0] = suma
            if k == self.n_epoch - 1:
                sumy[1] = suma
        self.coef_ = bs
        return sumy
    # stochastic gradient descent implementation


    def fit_mse(self, X_train, y_train):
        # initialized weights
        bs = [0,0,0,0]
        sumy = [[],[]]
        for k in range(self.n_epoch):
            suma = []
            for i, row in enumerate(X_train):
                coef_ = np.array(bs)
                dev = 0
                y_hat = self.predict_proba(row, coef_)
                suma.append((y_hat - y_train[i]) * (y_hat - y_train[i]) / len(X_train))
                for j in range(len(self.coef_)):
                    if j != 0:
                        dev = self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * row[j-1]
                    else:
                        dev = self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                    bs[j] -= dev
            if k == 0:
                sumy[0] = suma
            if k == self.n_epoch - 1:
                sumy[1] = suma
        self.coef_1 = bs
        return sumy

    def predict(self, X_test, cut_off=0.5):
        y_hat = [0 for _ in range(len(X_test))]
        y_hat1 = [0 for _ in range(len(X_test))]
        i = 0
        for row in X_test:
            y_hat1[i] = self.predict_proba(row,self.coef_1)
            y_hat[i] = self.predict_proba(row, self.coef_)
            i += 1
        for j in range(len(X_test)):
            if y_hat1[j] >= cut_off:
                y_hat1[j] = 1
            else:
                y_hat1[j] = 0
            if y_hat[j] >= cut_off:
                y_hat[j] = 1
            else:
                y_hat[j] = 0
        return np.array(y_hat),np.array(y_hat1)  # predictions are binary values - 0 or 1

b = [0.77001597, -2.12842434, -2.39305793]
data1 = load_breast_cancer(as_frame=True)
data11 = data1.frame['worst concave points']
data12 = data1.frame['worst perimeter']
data13 = data1.frame['worst radius']
X_sc, y = pd.DataFrame({'d1':np.array(data11),'d2':np.array(data12),'d3':np.array(data13)}), data1.frame['target']

st = StandardScaler()
st.fit(X_sc)
X = st.transform(X_sc)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
clr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
model = LogisticRegression()

err1 = clr.fit_log_loss(X_train,y_train)
err = clr.fit_mse(X_train,y_train)
model.fit(X_train,y_train)
y_pred, y_pred1 = clr.predict(X_test)
y_pred2 = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
acc1 = accuracy_score(y_test,y_pred1)
acc2 = accuracy_score(y_test,y_pred2)
