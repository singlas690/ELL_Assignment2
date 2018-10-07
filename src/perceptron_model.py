import numpy as np 

class perceptron_model:
    # Default euclidean distance with 5 neighbours
    def __init__(self, e= 1 , num_classes = 2, learning =0.01):
        self.e = e
        self.learning = learning
        self.num_classes = num_classes
        self.X = None
        self.Y = None
        self.weights = None

    # training data to X and Y
    def train(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        weights = np.zeros([self.num_classes, len(X_train[0])])
        for k in range(self.num_classes):
            print(k)
            for j in range(self.e):
                for i in range(len(X_train)):
                    result = np.dot(X_train[i],weights[k])
                    if result>=0 and Y_train[i] !=k:
                        weights[k] = np.subtract( np.transpose(weights[k]),self.learning*X_train[i])
                    elif result<0 and Y_train[i] ==k:
                        weights[k] = np.add(weights[k],self.learning*X_train[i])
        return weights

    def predict(self, X_test, weights):
        result = np.zeros([len(X_test),1])
        for c in range(len(X_test)):
            maxy =-1000000000000
            classy =-1
            for k in range(self.num_classes): 
                if np.dot(X_test[c],weights[k])>maxy:
                    classy = k 
                    maxy = np.dot(X_test[c],weights[k])
            result[c] = classy
        return result
