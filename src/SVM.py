import numpy as np 

class SVM:
    
    # Default euclidean distance with 5 neighbours
    def __init__(self, e= 1 , num_classes = 2, learn =0.1):
        #use constructor as  " perceptron_model(x,y,z) "
        self.e = e
        self.learning = learn
        self.num_classes = num_classes
        self.X = None
        self.Y = None
        self.weights = None

    # training data to X and Y
    def train(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        weights = np.zeros([self.num_classes, len(X_train[0])])
        
        epochs = self.e
        eta = self.learning
        errors = []
        
        for j in range(self.num_classes):
            for epoch in range(1,epochs):
                error = 0
                w = np.zeros(len(X_train[0]))
                for i, x in enumerate(X_train):
                    mult=-1
                    if Y_train[i]==j:
                        mult = 1
                    if (mult*np.dot(X_train[i], w)) < 1:
                        w = w + eta * ( (X_train[i] * Y_train[i]) + (-2  *(1/epoch)* w) )
                        error = 1
                    else:
                        w = w + eta * (-2  *(1/epoch)* w)
                    errors.append(error)
            weights[j] = w
        return weights
    

    def predict(self, X_test, weights):
        result = np.zeros([len(X_test)])
        for c in range(len(X_test)):
            maxy =-1000000000000000
            classy =-1
            
            for k in range(self.num_classes): 
                if np.dot(X_test[c],weights[k])>maxy:
                    classy = k 
                    maxy = np.dot(X_test[c],weights[k])
            result[c] = classy
        return result
    
if __name__ == "__main__":
    percy = SVM(1, 11, 0.11)
    print('xsy')
