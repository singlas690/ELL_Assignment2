import numpy as np 


def my_print(x):
	for i in range(x.shape[2]):
		print(x[:,:,i])

# Theory reference - http://www.cs.haifa.ac.il/~rita/ml_course/lectures_old/9_ParzenWin.pdf
class parzen_window:

	def __init__(self, window_type = 'hypercube', h = 1, dimension = 1, div_by_vol = True):
		self.window_type = window_type
		self.h = h
		self.dim = dimension
		self.div_by_volume = div_by_vol
	
	def estimate_parameters(self, X_train):
		self.X = X_train	

	# X - [m x dim x 1]
	def _hypercube_kernel_estimation(self, X_test):
		bool_all_dim = (np.abs(self.X - X_test)) <= self.h / 2
		num_points = np.sum(np.all(bool_all_dim, axis = 1, keepdims = True), axis = 0, keepdims = True)
		return num_points

	# X - [m x dim x 1]
	# assuming diaganol covariance
	def _gaussian_kernel_estimation(self, X_test):
		# print(X_test)
		exp_term = np.sum(-0.5 * np.power((self.X - X_test)/ self.h , 2) , axis = 1)
		p = (1/np.power(2*np.pi, self.dim/2)) * (1/np.power(self.h, self.dim)) * np.exp(exp_term)
		# print(p)
		num_points = np.sum(p, axis = 0)
		# print(num_points)
		return num_points
	
	def _distance(self, X):
		if (self.window_type == 'hypercube'):
			return self._hypercube_kernel_estimation(X)
		elif (self.window_type == 'gaussian'):
			return self._gaussian_kernel_estimation(X)

	# Returns p(X | theta) for a m test examples of dimension dim each - shape [m x 1]
	def get_likelihood(self, X_test):
		num_points = np.zeros((1, X_test.shape[0]))
		for i in range(X_test.shape[0]):
			num_points[0, i] = self._distance(X_test[i,:])
		# print(num_points)
		num_points = num_points / ((self.X).shape[0])
		if self.div_by_volume:
			num_points = num_points / self.h**self.dim
		return num_points.T

if __name__ == "__main__":

	print("Test parzen \n")
	train_s = 100
	test_s = 10
	dimt = 3

	X = 100 * np.random.randn(train_s,dimt) + 1 + 30 * np.random.randn(train_s,dimt) -1
	Xtest = 20 * np.random.randn(test_s,dimt) + 1
	
	estimator1 = parzen_window(dimension = dimt, h = 100)
	estimator1.estimate_parameters(X)

	
	print(estimator1.get_likelihood(Xtest))


	estimator2 = parzen_window(dimension = dimt, h = 10, window_type = 'gaussian',  div_by_vol = False)
	estimator2.estimate_parameters(X)

	
	print(estimator2.get_likelihood(Xtest))