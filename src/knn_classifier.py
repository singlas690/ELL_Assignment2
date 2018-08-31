import numpy as np 

class knn_model:

	# Default euclidean distance with 5 neighbours
	def __init__(self, k = 5, p = 2):
		self.k = k
		self.p = p
		self.X = None
		self.Y = None

	# Returns mode of each column, if no majority element is found, returns 0 indexed label (nearest neighbour)
	# Given [k x l] array , k neighbours of l test data points
	def _mode(self, labels):
		num_labels = np.max(labels)
		answer = np.zeros((1, labels.shape[1]))
		counts = np.zeros((1, labels.shape[1]))
		for i in range(num_labels):
			i_counts = np.sum(labels == i, axis = 0, keepdims = True)
			answer[(i_counts > counts)] = i
			counts = np.maximum(i_counts, counts) 
		return np.where([counts == 1], labels[0,:], answer)[0,:,:]

	# Returns minkowski distance between test vector and training data X with parameter self.p - [m x 1]
	# Given a numpy vector a of shape [1 x n] and self.X of shape [m x n] where m is number of training examples, n is dimension
	def _distance(self, a):
		if self.p == np.inf:
			return np.max(np.absolute(self.X - a), axis = 1)
		elif self.p == -np.inf:
			return np.min(np.absolute(self.X - a), axis = 1)
		else:
			return np.power(np.sum(np.power(np.absolute(self.X - a), self.p) , axis = 1), 1/self.p)

	# Assign training data to X and Y
	def train(self, X_train, Y_train):
		self.X = X_train
		self.Y = Y_train

	# Returns predicted labels Y - [l x 1] numpy array
	# Given m test examples of dimension n each X - [l x n] numpy array 
	def predict(self, X_test):
		dist = np.zeros(((self.X).shape[0], X_test.shape[0]))
		for i in range(X_test.shape[0]):
			dist[:, i] = self._distance(X_test[i,:])
		sorted_idx = np.argsort(dist, axis = 0)
		Y_per_test = np.repeat(self.Y, X_test.shape[0], axis = 1)
		k_nearest_labels = (Y_per_test[sorted_idx, np.arange(dist.shape[1])[np.newaxis, :]])[:self.k, :]
		
		return (self._mode(k_nearest_labels)).T

if __name__ == "__main__":
	print("Test KNN Class\n")

	x_train = (100*np.random.rand(10, 2)).astype('int')
	y_train = (4*np.random.rand(10, 1)).astype('int')

	x_test = (100*np.random.rand(4, 2)).astype('int')
	
	model1 = knn_model(k = 4)
	model1.train(x_train, y_train)
	predictions = model1.predict(x_test)

	print(np.hstack((x_train, y_train)))
	print(np.hstack((x_test, predictions)))
