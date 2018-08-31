import numpy as np 

# Theory reference - https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
def gaussian_mixture_model:

	# Structure reference - http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.__init__
	def __init__(self, n_comp = 1, tol = 0.001, max_iter = 1000, dimension = 1):
		self.n_comp = n
		self.tol = tol
		self.max_iter = max_iter
		self.dim = dimension
		self.mean_vector = None
		self.covariance = None
		self.alpha = None
		self.weights = None

	def initialize(self):
		# Initialize mu of dim dimension in range [-1, 1] for each component
		self.mean_vector = 2 * np.random.rand(self.dim, self.n_comp) - 1
		# Initialize covariance matrix of [dim x dim] dimension in range [0, 1] for each component
		self.covariance = np.random.rand(self.dim, self.dim, self.n_comp)
		# Initialize alpha - [k x 1] matrix with sum of elements = 1
		temp = np.random.rand(self.n_comp, 1)
		self.alpha = temp/np.sum(temp)

	def estimate(self, X_train):
