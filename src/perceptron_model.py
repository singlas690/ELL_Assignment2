import numpy as np 

class perceptron_model:
    
    def __init__(self, iterations= 1, num_classes = 2, learn = 1):
        #use constructor as  " perceptron_model(x,y,z) "
        self.iter = iterations
        self.learning = learn
        self.num_classes = num_classes
        self.X = None
        self.Y = None
        self.weight = None

    # training data to X and Y
    def fit(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        weights = np.zeros([self.num_classes, len(X_train[0])])
        for k in range(self.num_classes):
            for j in range(self.iter):
                for i in range(len(X_train)):
                    result = np.dot(X_train[i], weights[k])
                    if result >= 0 and Y_train[i] != k:
                        weights[k] = np.subtract(np.transpose(weights[k]), self.learning*X_train[i])
                    elif result<0 and Y_train[i] == k:
                        weights[k] = np.add(weights[k], self.learning*X_train[i])
        
        self.weight = weights

    def predict(self, X_test):
        result = np.zeros([len(X_test)])
        for c in range(len(X_test)):
            maxy =-1000000000000
            class_y = -1
            for k in range(self.num_classes): 
                if np.dot(X_test[c], self.weight[k]) > maxy:
                    class_y = k 
                    maxy = np.dot(X_test[c], self.weight[k])
            result[c] = class_y
        return result.reshape((len(X_test), 1))
    
if __name__ == "__main__":
    percy = perceptron_model(iterations = 1, num_classes = 5, learn = 1)
    X_train = (100*np.random.rand(10, 3)).astype('int')
    Y_train = (5*np.random.rand(10, 1)).astype('int')
    X_test = (100*np.random.rand(4, 3)).astype('int')
    percy.fit(X_train, Y_train)
    Y_pred = percy.predict(X_test)
    print(Y_pred.shape)