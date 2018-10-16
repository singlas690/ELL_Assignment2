import numpy as np
import scipy.sparse
import pandas as pd
class softmaxLogit:

    # Use this for multiclass classification using logistic classifer after one hot
    # encoding of y vector
    # One hot encoding can be done using helper func defined below or using pandas.dummies      

    def __init__(self, Lambda= 1, alpha = 0.001, iterations = 10000, num_classes = 2):
        self.Lambda = Lambda
        self.alpha = alpha
        self.num_iter = iterations
        self.weight = None
        self.classes = num_classes
        
    def getLoss(self, w, x, y, lam):
        m = x.shape[0]                      #First we get the number of training examples
        y_mat = self.oneHotIt(y)            #Next we convert the integer class coding into a one-hot representation
        scores = np.dot(x, w)               #Then we compute raw class scores given our input and current weights
        prob = self.softmax(scores)         #Next we perform a softmax on these scores to get their probabilities
        loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w)             #We then find the loss of the probabilities
        grad = (-1 / m) * np.dot(x.T, (y_mat - prob)) + lam*w                            #And compute the gradient for that loss
        return loss, grad
    
    # Either use this helper func or pandas.dummies for one hot encoding of y
    def oneHotIt(self, Y):
        return np.squeeze(np.eye(self.classes)[Y.reshape(-1)])
#         m = len(Y)
#         b = np.zeros((m, self.classes))
#         b[np.arange(3), a] = 1
#         return b
    
    def softmax(self, z):
        z -= np.max(z)
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
        return sm

    ### main loop
    
    def fit(self, x, y):    
        w = np.zeros([x.shape[1], len(np.unique(y))])
        lam = self.Lambda
        iterations = self.num_iter
        learningRate = self.alpha
        losses = []
        for i in range(0, iterations):
            loss,grad = self.getLoss(w, x, y, lam)
            losses.append(loss)
            w = w - (learningRate * grad)
        self.weight = w
    
    def predict(self, someX):
        probs = self.softmax(np.dot(someX, self.weight))
        preds = np.argmax(probs, axis=1)
        return preds.reshape((len(preds), 1))
    
    def predict_proba(self, someX):
        probs = self.softmax(np.dot(someX, self.weight))
        return probs


    def getAccuracy(self, someX, someY):
        # caution : this method takes X and Y as input and not y_predicted and y_actual
        
        prob, prede = self.predict(someX)
        accuracy = sum(prede == someY)/(float(len(someY)))
        return accuracy
