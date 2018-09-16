import numpy as np 


def my_print(x):
	for i in range(x.shape[2]):
		print(x[:,:,i])

# Theory reference - http://www.cs.haifa.ac.il/~rita/ml_course/lectures_old/9_ParzenWin.pdf
class parzen_window:

	# Structure reference - http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.__init__
	def __init__(self, center, window_type = 'hypercube', h, dimension):
		self.window_type = window_type
		self.h = h
		self.dim = dimension
	
	def estimate_parameters(self, X_train):
		self.X = X_train	

	def _distance(self, X):
		if (self.window_type == 'hypercube'):
			return _hypercube_kernel_estimation(X)
		else if (self.window_type == 'gaussian'):
			return _gaussian_kernel_estimation(X)

	# X - [m x dim x 1]
	def _hypercube_kernel_estimation(self, X_test):
		bool_all_dim = (np.abs(self.X - X_test)) <= self.h / 2
		num_points = np.sum(np.all(bool_all_dim, axis = 1, keepdims = True), axis = 0, keepdims = True)
		return num_points

	# X - [m x dim x 1]
	# assuming diaganol covariance
	def _gaussian_kernel_estimation(self, X_test):
		exp_term = -0.5 * (np.power( (self.X - X_test)/ self.h), 2)
		p = (1/np.power(2*np.pi, self.dim/2)) * (1/np.power(self.h, d)) * np.exp(exp_term)
		num_points = np.sum(p, axis = 0, keepdims = True)
		return num_points
	
	# Returns p(X | theta) for a m test examples of dimension dim each - shape [m x 1]
	def get_likelihood(self, X_test):
		num_points = np.zeros((1, X_test.shape[0]))
		for i in range(X_test.shape[0]):
			num_points[0, i] = self._distance(X_test[i,:])
		num_points = num_points / ((self.X).shape[0] * self.h**2 )
		return num_points.T
