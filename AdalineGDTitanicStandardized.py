import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.colors import ListedColormap
from sklearn import preprocessing

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """ Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, 
            where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
    
        Returns
        -------
        self : object
    
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
    
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    

def print_full(x):
   pd.set_option('display.max_rows', len(x))
   print(x)
   pd.reset_option('display.max_rows')
   
def main():
    df = pd.read_csv('titanic.csv')
    df = df.dropna(subset =['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    df.apply(pd.to_numeric, errors='ignore')
  
    y = df.iloc[0:500, 1].values
    y = np.where(y == 0, -1, 1)
    X = df.iloc[0:500, [2, 4, 5, 6, 7, 9]].values
    X[X == 'male'] = 0
    X[X == 'female'] = 1
    X_proc = preprocessing.scale(X)

    
    ada = AdalineGD(eta=0.00001, n_iter=100)
    ada.fit(X_proc, y)
    X_sub = df.iloc[501:714, [2, 4, 5, 6, 7, 9]].values
    X_sub[X_sub == 'male'] = 0
    X_sub[X_sub == 'female'] = 1
    X_subproc = preprocessing.scale(X_sub)
    y_sub = df.iloc[501:714, 1].values
    y_sub = np.where(y_sub == 0, -1, 1)
    X_result = ada.predict(X_subproc)
    results = np.equal(X_result, y_sub)
    correct = np.sum(results)
    accuracy = (correct / 213) *100
    print(str(accuracy) + '% Accuracy')
    print(ada.w_)
    
    
    


main()