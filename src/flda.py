import numpy as np

class flda:

	def __init__(self, num_classes = 2):
		self.num_classes = num_classes
		self.X = None
		self.Y = None
		self.weight = None

	def _find_mean(self, X_train, Y_train, k):
		n, m = X_train.shape
		mean0 = mean1 = n_prime = 0		

		for i in range(n):
			if (Y_train[i]==k):
				mean1 = mean1 + X_train(i)
			else:
				mean0 = mean0 + x_train(i)

		mean1 = mean1/n_prime
		mean0 = mean0/(n-n_prime)
		return [mean0, mean1]

	def _find_sw(self, X_train, Y_train, mean0, mean1, k):
		n, m = X_train.shape
		sw = 0

		for i in range(n):
			if (Y_train[i]==k):
				sw = sw + (X_train(i) - mean1)*(X_train(i) - mean1).T
			else:
				sw = sw + (X_train(i) - mean0)*(X_train(i) - mean0).T
		return sw
	
	def train(self, X_train, Y_train):
		self.X = X_train
		self.Y = Y_train

		for k in range(self.num_classes):
			Y_prime = [Y_train[labels==k] for labels in range(Y_train.shape[0])]
			[mean0, mean1] = self._find_mean(X_train, Y_train, k)
			sw = self._find_sw(X_train, Y_train, mean0, mean1, k)

		wgt = np.matmul(np.linalg.inv(sw), (mean1 - mean0))
		self.weight.append(wgt)

	def predict(self, X_test):
		Y_test = np.zeros([len(X_test),1])
		for c in range(len(X_test)):
			maxy = -1000000000000
			for k in range(self.num_classes): 
				if np.dot(X_test[c],weight[k]) > maxy:
					class_pred = k 
					maxy = np.dot(X_test[c],weights[k])
			result[c] = class_pred
		return result

if __name__ == '__main__':
	X_train = (100*np.random.rand(10, 3)).astype('int')
	Y_train = (5*np.random.rand(10, 1)).astype('int')
	X_test = (100*np.random.rand(4, 3)).astype('int')
	percy = flda(num_classes = 5)
	percy.train(X_train, Y_train)
	Y = percy.predict(X_test)