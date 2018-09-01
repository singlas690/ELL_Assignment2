import numpy as np 

def my_print(x):
	for i in range(x.shape[2]):
		print(x[:,:,i])

# Theory reference - https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
class gaussian_mixture_model:

	# Structure reference - http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.__init__
	def __init__(self, n_comp = 1, tol = 0.001, max_iter = 1000, dimension = 1):
		self.n_comp = n_comp
		self.tol = tol
		self.max_iter = max_iter
		self.dim = dimension
		self.mean_vector = None
		self.covariance = None
		self.alpha = None
		self.weights = None

	def initialize(self):
		# Initialize mu of dim dimension in range [-1, 1] for each component - shape [ 1 x dim x n_comp]
		self.mean_vector = 2 * np.random.rand(1, self.dim, self.n_comp)
		# Initialize covariance matrix of [dim x dim] dimension in range [0, 1] for each component - shape [ dim x dim x n_comp]
		self.covariance = np.random.rand(self.dim, self.dim, self.n_comp)
		# Initialize alpha - [k x 1] matrix with sum of elements = 1
		temp = np.random.rand(self.n_comp, 1)
		self.alpha = temp/np.sum(temp)
		# my_print(self.mean_vector)
		# print()
		# my_print(self.covariance)
		# print()
		# print(self.alpha)
		# print()

	def estimate_parameters(self, X_train):
		# Initialize weights of n training examples for each component - shape [ n x n_comp]
		self.weights = np.zeros((X_train.shape[0], self.n_comp))
		# Reshape X to [n x dim x 1]
		my_print(np.expand_dims(X_train, axis = 2))
		print()		
		self._em(np.expand_dims(X_train, axis = 2))

		return [self.mean_vector, self.covariance, self.alpha]

	# GOD MODE ACTIVATED
	def _em(self, X):
		num_iter = 0
		error = 10 * self.tol
		while error > self.tol and num_iter < self.max_iter :
			num_iter += 1
			print(num_iter)
			old_mean = self.mean_vector
			old_covariance = self.covariance
			old_alpha = self.alpha

			self._expectation(X)
			self._maximization(X)

			error = max(np.max(np.abs(old_alpha - self.alpha)), np.max(np.abs(old_mean - self.mean_vector)),
					np.max(np.abs(old_covariance - self.covariance)))
			print(error)
	def _expectation(self, X):
		numerator = self._probability(X) * self.alpha.T
		self.weights = numerator/np.sum(numerator, axis = 1, keepdims = True)

	def _maximization(self, X):
		self.alpha = np.sum(self.weights, axis = 0, keepdims = True).T / X.shape[0]
		
		mean_numerator = np.sum(np.expand_dims(self.weights, axis = 1) * X , axis = 0, keepdims = True)
		self.mean_vector = mean_numerator / np.expand_dims(np.sum(self.weights, axis = 0, keepdims = True) , axis = 0)

		X_u = X - self.mean_vector
		cov_numerator = np.einsum('ikl,kjl->ijl', X_u.transpose(1 , 0, 2), X_u * np.expand_dims(self.weights, axis = 1))
		self.covariance = cov_numerator / np.expand_dims(np.sum(self.weights, axis = 0, keepdims = True) , axis = 0)


	def _probability(self, X):
		# *_* WOW!!!!
		# https://stackoverflow.com/questions/41850712/compute-inverse-of-2d-arrays-along-the-third-axis-in-a-3d-array-without-loops
		cov_inverse = np.linalg.inv(self.covariance.T).T
		X_u = X - self.mean_vector
		# my_print(cov_inverse)
		# print()
		# my_print(X_u)
		# print()
		# t = np.einsum('ikl,kjl->ijl', X_u, cov_inverse)
		# print(all(np.allclose(t[:,:,i], (X_u[:,:,i] @ cov_inverse[:,:,i])) for i in range(2)))
		exp_term = -0.5 * (np.einsum('ikl,kjl->ijl', np.einsum('ikl,kjl->ijl', X_u, cov_inverse), 
					X_u.transpose((1,0,2)))).diagonal(0,0,1).T
		# print(exp_term)
		# print()
		p = (1/np.power(2*np.pi, self.dim/2)) * (1/np.linalg.det(self.covariance.T).reshape(1,self.n_comp)) * np.exp(exp_term)
		return p

if __name__ == "__main__":
	print("Test GMM \n")

	X = 2.5 * np.random.randn(5,3) + 3
	
	estimator1 = gaussian_mixture_model(n_comp = 2, dimension = 3)

	# How to initialize
	# Sometimes floating point approximation of small numbers to 0, division by 0 
	estimator1.initialize()
	estimator1.estimate_parameters(X)

	print(estimator1.mean_vector)