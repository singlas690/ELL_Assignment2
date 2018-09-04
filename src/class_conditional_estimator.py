import numpy as np 
from sklearn.mixture import GaussianMixture

def my_print(x):
	for i in range(x.shape[2]):
		print(x[:,:,i])

# Theory reference - https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
class gaussian_mixture_model:

	# Structure reference - http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.__init__
	def __init__(self, n_comp = 1, tol = 0.001, max_iter = 1000, dimension = 1, regularize = 0.00001):
		self.n_comp = n_comp
		self.tol = tol
		self.max_iter = max_iter
		self.dim = dimension
		self.reg = regularize

		# Initialize mu of dim dimension as 0 for each component - shape [ 1 x dim x n_comp]
		self.mean_vector = np.zeros((1, self.dim, self.n_comp))

		# Initialize covariance matrix of [dim x dim] dimension as 0 for each component - shape [ dim x dim x n_comp]
		self.covariance = np.zeros((self.dim, self.dim, self.n_comp))

		# Initialize alpha - [k x 1] matrix as 0
		self.alpha = np.zeros((self.n_comp, 1))

		self.weights = None
	
	def estimate_parameters(self, X_train):
		# Initialize weights of n training examples for each component - shape [ n x n_comp]
		self.weights = np.random.rand(X_train.shape[0], self.n_comp)
		# Reshape X to [n x dim x 1]
		self._em(np.expand_dims(X_train, axis = 2))

		return [self.mean_vector, self.covariance, self.alpha]

	# GOD MODE ACTIVATED

	# Returns p(X | theta) for a m test examples of dimension dim each - shape [m x dim]
	def get_likelihood(self, X_test):
		# return self._probability(np.expand_dims(X_test, axis = 2))
		return np.sum(self._probability(np.expand_dims(X_test, axis = 2)) * self.alpha.T , axis = 1, keepdims = True)

	# EM algorithm
	def _em(self, X):
		num_iter = 0
		error = 10 * self.tol
		old_likelihood = np.inf
		while error > self.tol and num_iter < self.max_iter :
			num_iter += 1
			
			self._maximization(X)
			self._expectation(X)
		
			error = abs(old_likelihood - self._likelihood(X)[0,0])
			old_likelihood = self._likelihood(X)[0,0]
	
	# Returns Data Likelihood
	def _likelihood(self, X):
		return np.sum(np.log(np.sum(self._probability(X) * self.alpha.T , axis = 1, keepdims = True)), axis = 0, keepdims = True)

	# E Step
	def _expectation(self, X):
		numerator = self._probability(X) * self.alpha.T
		self.weights = numerator/np.sum(numerator, axis = 1, keepdims = True)
	
	# M Step
	def _maximization(self, X):
		self.alpha = np.sum(self.weights, axis = 0, keepdims = True).T / X.shape[0]
		
		mean_numerator = np.sum(np.expand_dims(self.weights, axis = 1) * X , axis = 0, keepdims = True)
		self.mean_vector = mean_numerator / np.expand_dims(np.sum(self.weights, axis = 0, keepdims = True) , axis = 0)
		
		X_u = X - self.mean_vector
		cov_numerator = np.einsum('ikl,kjl->ijl', X_u.transpose(1 , 0, 2), X_u * np.expand_dims(self.weights, axis = 1))
		# Regularization Idea - https://stats.stackexchange.com/questions/35515/matlab-gmdistribution-fit-regularize-what-regularization-method
		self.covariance = cov_numerator / np.expand_dims(np.sum(self.weights, axis = 0, keepdims = True) , axis = 0) + np.expand_dims(np.diag(np.diag(np.ones((self.dim,self.dim)))), axis = 2) * self.reg
	
	# Given X as n examples of dimension d ([n x dim x 1]), returns pdf value for each example for each gmm component - shape [n x k]
	def _probability(self, X):
		# *_* WOW!!!!
		# https://stackoverflow.com/questions/41850712/compute-inverse-of-2d-arrays-along-the-third-axis-in-a-3d-array-without-loops
		cov_inverse = np.linalg.inv(self.covariance.T).T
		X_u = X - self.mean_vector
		
		exp_term = -0.5 * (np.einsum('ikl,kjl->ijl', np.einsum('ikl,kjl->ijl', X_u, cov_inverse), 
					X_u.transpose((1,0,2)))).diagonal(0,0,1).T
		
		p = (1/np.power(2*np.pi, self.dim/2)) * (1/np.sqrt(np.abs(np.linalg.det(self.covariance.T).reshape(1,self.n_comp)))) \
		* np.exp(exp_term)
		
		return p

if __name__ == "__main__":
	print("Test GMM \n")
	train_s = 100
	test_s = 2
	dimt = 3
	comp = 2
	X = 2 * np.random.randn(train_s,dimt) + 1 + 3 * np.random.randn(train_s,dimt) -1
	Xtest = 2 * np.random.randn(test_s,dimt) + 1
	
	skmodel = GaussianMixture(n_components = comp,  init_params='random')
	estimator1 = gaussian_mixture_model(n_comp = comp, dimension = dimt)

	skmodel.fit(X)

	print(skmodel.score(X))

	# How to initialize
	estimator1.estimate_parameters(X)

	print(estimator1._likelihood(np.expand_dims(X, axis = 2)) / train_s)
	
	print()
	print(skmodel.weights_)
	print(estimator1.alpha)
	print()
	print(estimator1.get_likelihood(Xtest))
	print(np.sum(np.exp(skmodel._estimate_weighted_log_prob(Xtest)) * skmodel.weights_ , axis = 1) )
	print()
	print(estimator1._likelihood(np.expand_dims(Xtest, axis = 2)) / test_s)
	print(skmodel.score(Xtest))
