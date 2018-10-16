#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:16:15 2018

@author: harsh
"""

import numpy as np
from numpy.linalg import inv

class LinearModel:
    
    
    def __init__(self, solver = 'gd', Lambda = 0, num_iter = 100000, alpha = .01):
        # Two options for solver: 'gd' or 'ols'
        self.X = None
        self.y = None
        self.Lambda = Lambda                 # regularisation constant
        self.solver = solver                # type of solver: 'ols' or 'gd'
        self.coeff = None                 # coeffiecients of the hypothesis
        self.num_iter = num_iter   # no of iterations for gradient descent
        self.alpha = alpha               # step size for gradient descent
        self.cost_history = None
        
    def cost_func(self, X, y, theta, Lambda):
        m = len(y)
        J = 0
        grad = np.zeros(theta.shape)
        #print('Shape grad 1', grad.shape)
        temp = theta
        temp[0] = 0
        J = (sum(np.square(X.dot(theta) - y)) + (Lambda/(2*m))*sum(np.square(temp)));
        grad = ((X.T.dot(X.dot(theta)-y)) + (Lambda*temp))/m;
        #print('the shape of grad is', grad.shape)
        #print('the shape of J is ', J.shape)
        return (J , grad)
        
    
    def train(self, X_train, y_train):
        # y_train is a column vector (mX1)
        # X_train is 2d array (mXn)
        m = X_train.shape[0]
        n = X_train.shape[1]
        ones = np.ones((1, m))
        X_aug = np.concatenate((ones, X_train.T), axis = 0).T
        self.X = X_aug
        self.y= y_train.reshape((m, 1))
        #self.coeff = np.zeros((self.X.shape[1],1))
        if self.solver=='ols':
            self.coeff = inv(self.Lambda*np.identity(n) + X_aug.transpose().dot(X_aug)).dot(X_aug.transpose()).dot(y_train)
        if self.solver=='gd':
            a = self.gradient_descent(self.X, self.y, self.Lambda, self.alpha, self.num_iter)
            self.coeff = a[0]
            self.cost_history = a[1]
            
        
    def gradient_descent(self, X,y, Lambda, alpha, iterations):
    
        cost_history = [0] * iterations
        
        theta = np.zeros((X.shape[1], 1))
        #print('running the gradient descent function\n')
        #print('the shape of theta is', theta.shape)
    
        for iteration in range(iterations):
            
            gradient = self.cost_func(X, y, theta, Lambda)[1]
            #print('the shape of gradient at line 69', gradient.shape)
            # Changing Values of theta using Gradient
            theta = theta - alpha * gradient
            #print('the shape of updated coeff in line 72',self.coeff.shape)
            # New Cost Value
            cost = self.cost_func(X, y, theta, Lambda)[0]
            cost_history[iteration] = cost
        
        return (theta, cost_history)

        
    
    def predict(self, X_test):
        m = X_test.shape[0]
        ones = np.ones((1,m))
        X_aug_test = np.concatenate((ones, X_test.T), axis = 0).T
        y_pred = X_aug_test.dot(self.coeff)
        return y_pred
    
    def _get_params(self):
        pass
    
    
if __name__ == "__main__":

    print("Test Linear Model class\n")
    '''
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()


    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
'''

    X_train = 10*np.random.randn(100,1)+5
    y_train = 3*X_train +1
    # Create linear regression object
    #regr = linear_model.LinearRegression()
    regr = LinearModel(solver='gd')
    # Train the model using the training sets
    #regr.fit(diabetes_X_train, diabetes_y_train)
    regr.train(X_train, y_train)
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(X_train)
    
    # The coefficients
    print('Coefficients: \n', regr.coeff)
    # The mean squared error
    print("Mean squared error: %.2f"
      % mean_squared_error(y_train, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_train, diabetes_y_pred))

    # Plot outputs
    plt.scatter(X_train, y_train,  color='black')
    plt.plot(X_train, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    
        

