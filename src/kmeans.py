# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:37:16 2018

@author: pc
"""
import numpy as np
class kmeans:
    
    # Default no. of clusters = 3
    # Defualt cluster centroids to be chosen randomly from training data
    #def __init__(self, n_clusters = 3, rand_centroid = True, max_iter = 100):
    def __init__(self, n_clusters = 3, max_iter = 100):        
        self.n_clusters = n_clusters
        self.centroids = None
        self.train_labels = None
        self.max_iter = max_iter
        self.X = None
        
    # Initialize random cluster centroids: choosing the cluster centres randomly from data set
    def _rand_centroid(self, X_train):
        c = np.random.permutation(X_train)
        centroids = np.zeros((self.n_clusters,X_train.shape[1]))
        for i in range(0, self.n_clusters):
            centroids[i,:]= c[i,:]
        return centroids
            
        
        
    # func to calculate closest centroids
    # takes X_train [m X n] and centroids [n_clusters X n] 
    # returns idx[m X 1] containing the index of nearest cluster of each data point
    def _closest_centroid(self, X_train, centroids):
        num_centroids , dim = centroids.shape
        num_points , _ = X_train.shape
        # Tile and reshape both arrays into `[num_points, num_centroids, dim]`.
        centroids = np.tile(centroids, [num_points, 1]).reshape([num_points, num_centroids, dim])
        X_train = np.tile(X_train, [1, num_centroids]).reshape([num_points, num_centroids, dim])

        # Compute all distances (for all points and all centroids) at once and 
        # select the min centroid for each point.
        distances = np.sum(np.square(centroids - X_train), axis=2)
        return np.argmin(distances, axis=1)
                
                

    # func to return new centroid means
    # given the data and respective labels
    def _updated_means(self, X_train, labels):
        centroids = np.zeros((self.n_clusters, X_train.shape[1]))
        return np.array([X_train[labels==k].mean(axis=0) for k in range(centroids.shape[0])])
    
    
    # train method to assign X [m X n] training data
    # GIVES THE CLUSTER CENTROIDS AND CLUSTER LABELS FOR THE TRAIN DATA
    def train(self, X_train):
        self.X = X_train
        centroids = self._rand_centroid(X_train)
        for i in range(0, self.max_iter):
            labels = self._closest_centroid(X_train,centroids)
            centroids = self._updated_means(X_train, labels)
        self.centroids = centroids
        self.train_labels = labels
        
        
        
    # predict method to run kmeans algo 
    def predict(self, X_test):
        test_labels = self._closest_centroid(X_test, self.centroids)    
        return test_labels
    
    def get_centroids(self):
        return self.centroids
        
if __name__ == "__main__":
    print("Test K-means class\n")
    x_train = (100*np.random.rand(300, 2)).astype('int')
    x_test = (100*np.random.rand(20, 2)).astype('int')
    model1 = kmeans(2,20)
    model1.train(x_train)
    train_labels = model1.train_labels
    centroids = model1.centroids
    test_labels = model1.predict(x_test)
    
    from matplotlib import pyplot as plt
    import matplotlib
    colors = ['blue','green']
    # plt.scatter(x_train[:,0],x_train[:,1],c=train_labels,cmap=matplotlib.colors.ListedColormap(colors))
    plt.scatter(centroids[:,0],centroids[:,1],c='red')
    plt.scatter(x_test[:,0],x_test[:,1],c=test_labels,cmap=matplotlib.colors.ListedColormap(['black','purple']))
	
    plt.show()
	#print(np.hstack((x_train, y_train)))
	#print(np.hstack((x_test, predictions)))
            

    
