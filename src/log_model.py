    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:37:38 2018

@author: harsh
"""

import numpy as np

class LogitModel:
    

    def __init__(self, alpha=0.01, num_iter=100000, Lambda = 0, verbose = True):
        self.alpha = alpha
        self.num_iter = num_iter
        self.Lambda = Lambda
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def log_loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def train(self, X, y):

        intercept = np.ones((X.shape[0], 1))
        X_aug = np.concatenate((intercept, X), axis=1)
        
        # weights initialization
        self.theta = np.zeros(X_aug.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X_aug, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X_aug.T, (h - y)) / y.size + (self.Lambda/y.size)*self.theta
            gradient[0]=gradient[0]-(self.Lambda/y.size)*self.theta[0]
            self.theta -= self.alpha * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X_aug, self.theta)
                h = self.sigmoid(z)
                print('train loss after ',i,'iterations:', self.log_loss(h, y), '\t')
    
    def predict_prob(self, X):
        intercept = np.ones((X.shape[0], 1))
        X_aug = np.concatenate((intercept, X), axis=1)
    
        return self.sigmoid(np.dot(X_aug, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    

if __name__ == "__main__":
    

# example of training a final classification model
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit(X_train).transform(X_train)
    X_test_scaled = scaler.fit(X_test).transform(X_test)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    print('Accuracy on the training set: {:.3f}'.format(log_reg.score(X_train,y_train)))
    print('Accuracy on the training set: {:.3f}'.format(log_reg.score(X_test,y_test))) 
    
    my = LogitModel()
    my.train(X_train_scaled, y_train)
    y_pred = my.predict(X_train_scaled, 0.5)
    print('Accuracy on the training set: {:.3f}'.format(metrics.accuracy_score(y_pred,y_train)))
    print('Accuracy on the training set: {:.3f}'.format(metrics.accuracy_score(my.predict(X_test_scaled, 0.5),y_test)))     
