#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:52:34 2018

@author: harsh
"""

import numpy as np

class multinormal_model:
    
    def __init__(self, n_variables=1 ):
        self.n_variables = n_variables
        self.mean_vector = np.zeros((self.n_variables,))
        self.covariance = np.zeros((self.n_variables,self.n_variables))
        
        
    def estimate_parameters(self, X_train):
        self.n_variables= (X_train).shape[1]
        mu = np.divide(X_train.sum(axis=0),(X_train).shape[0])
        self.mean_vector = mu
        sum_matrix = self.covariance
        for i in range(0,(X_train).shape[0]):
            X_i = X_train[i,:]
            a=np.tensordot(np.subtract(X_i-self.mean_vector),np.subtract(X_i-self.mean_vector),axes=0)
            sum_matrix = np.add(a,sum_matrix)
        self.covariance= np.divide(sum_matrix, (X_train).shape[0])
        return [self.mean_vector, self.covariance]
        
        
    def get_likelihood(self, X_test):
        X_u = X_test - self.mean_vector
        exp_term = -0.5*np.diagonal(np.matmul(np.matmul(X_u,self.covariance),X_u.T)
        p = (1/np.power(2*np.pi, self.n_variables/2))* (1/np.sqrt(np.abs(np.linalg.det(self.covariance))))* np.exp(exp_term)
        return np.array([p]).T
        