import numpy as np
import random

class flda:

	def __init__(self, num_classes = 2):
		self.num_classes = num_classes
		self.X = None
		self.Y = None
		self.weight = []

	def _find_mean(self, X_train, Y_train, k):
		n, m = X_train.shape
		mean0 = mean1 = n_prime = 0		

		for i in range(n):
			if (Y_train[i]==k):
				mean1 = mean1 + X_train[i]
				n_prime += 1
			else:
				mean0 = mean0 + X_train[i]

		mean1 = mean1/n_prime
		mean0 = mean0/(n-n_prime)
		mean1 = mean1.reshape(1, m)
		mean0 = mean0.reshape(1, m)
		return [mean0, mean1]

	def _find_sw(self, X_train, Y_train, mean0, mean1, k):
		n, m = X_train.shape
		sw = 0
		for i in range(n):
			if (Y_train[i]==k):
				sw = sw + np.matmul((X_train[i].reshape(1, m) - mean1).T, (X_train[i].reshape(1, m) - mean1))
			else:
				sw = sw + np.matmul((X_train[i].reshape(1, m) - mean0).T, (X_train[i].reshape(1, m) - mean0))
		return sw
	
	def train(self, X_train, Y_train):
		self.X = X_train
		self.Y = Y_train

		for k in range(self.num_classes):
			total = np.count_nonzero(Y_train == k)
			if(total == 0):
				wgt = np.zeros((X_train.shape[1], 1))
			else:
				[mean0, mean1] = self._find_mean(X_train, Y_train, k)
				sw = self._find_sw(X_train, Y_train, mean0, mean1, k)
				wgt = np.matmul(np.linalg.inv(sw), (mean1 - mean0).T)
			self.weight.append(wgt)

	def predict(self, X_test):
		result = np.zeros([len(X_test),1])
		Y_test = np.zeros([len(X_test),1])
		for c in range(len(X_test)):
			maxy = -1000000000000
			for k in range(self.num_classes): 
				if np.dot(X_test[c], self.weight[k]) > maxy:
					class_pred = k 
					maxy = np.dot(X_test[c], self.weight[k])
			result[c] = class_pred
		return result

if __name__ == '__main__':
	X_train = (100*np.random.rand(10, 3)).astype('int')
	Y_train = (5*np.random.rand(10, 1)).astype('int')
	X_test = (100*np.random.rand(4, 3)).astype('int')
	percy = flda(num_classes = 5)
	percy.train(X_train, Y_train)
	Y = percy.predict(X_test)
