#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 23:49:18 2018

@author: harsh
"""

import numpy as np

class multinomial_mle:
    '''This is for the special case for dicrete valued features in Naive Bayes '''
    #def __init__(self, n_vals):
    def __init__(self):
        
        #no of discrete values in the feature vector
        #self.n_vals = n_vals
        self.proba = {}
        
    def estimate_parameters(self, X_train):
        # X_train is a discrete valued column vector
        unique, counts = np.unique(X_train, return_counts=True)
        prob = counts/len(X_train)
        self.proba=dict(zip(unique, prob))
        # returns a dict with probability associated with each discrete element
        return self.proba
        
    
    def get_likelihood(self, X_test):
        p=[]
        for i in X_test:
            if i in self.proba:
                p.append(self.proba[i])
            else:
                p.append(0)
        # returns a mX1 ndarray with prob corresponding to each point        
        return np.array([p]).T
