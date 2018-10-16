import numpy as np
import scipy.sparse
import pandas as pd
class softmaxLogit:

    # Use this for multiclass classification using logistic classifer after one hot
    # encoding of y vector
    # One hot encoding can be done using helper func defined below or using pandas.dummies      

    def __init__(self, Lambda= 1, alpha = 0.001, iterations = 10000):
        self.Lambda = Lambda
        self.alpha = alpha
        self.num_iter = iterations
        self.weight = None
        
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
        # m = Y.shape[0]
        # Y = Y[:,0]
        # OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        # OHX = np.array(OHX.todense()).T'''
        df = pd.DataFrame({'target':Y})
        df1 = pd.get_dummies(df, columns=["target"], drop_first=False)
        return df1.values       

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
        return preds
    
    def predict_proba(self, someX):
        probs = self.softmax(np.dot(someX, self.weight))
        return probs


    def getAccuracy(self, someX, someY):
        # caution : this method takes X and Y as input and not y_predicted and y_actual
        
        prob, prede = self.predict(someX)
        accuracy = sum(prede == someY)/(float(len(someY)))
        return accuracy


