import numpy as np 

def my_print(x):
	for i in range(x.shape[2]):
		print(x[:,:,i])

# Theory reference - https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
class gaussian_mixture_model:

	# Structure reference - http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.__init__
	def __init__(self, n_comp = 1, tol = 0.001, max_iter = 1000, dimension = 1, regularize = 0.00001, clipval = True):
		self.n_comp = n_comp
		self.tol = tol
		self.max_iter = max_iter
		self.dim = dimension
		self.reg = regularize
		self.clip = clipval

		# Initialize mu of dim dimension as 0 for each component - shape [ 1 x dim x n_comp]
		self.mean_vector = np.zeros((1, self.dim, self.n_comp))

		# Initialize covariance matrix of [dim x dim] dimension as 0 for each component - shape [ dim x dim x n_comp]
		self.covariance = np.zeros((self.dim, self.dim, self.n_comp))

		# Initialize alpha - [k x 1] matrix as 0
		self.alpha = np.zeros((self.n_comp, 1))

		self.weights = None
	
	def estimate_parameters(self, X_train):
		# Initialize weights of n training examples for each component - shape [ n x n_comp]
		temp = np.random.rand(X_train.shape[0], self.n_comp)
		self.weights = temp/np.sum(temp, axis = 1, keepdims = True)
		# print(self.weights)
		# self.weights = np.random.rand(X_train.shape[0], self.n_comp)
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
			# print(num_iter)	
			self._maximization(X)
			self._expectation(X)
			
			error = abs(old_likelihood - self._likelihood(X)[0,0])
			old_likelihood = self._likelihood(X)[0,0]

		if (num_iter == self.max_iter):
			print("Didn't converge, ran for %d iterations" % num_iter)
		else:
			print("Converged after %d iterations" % num_iter)
	
	# Returns Data Likelihood
	def _likelihood(self, X):
		return np.sum(np.log(np.sum(self._probability(X) * self.alpha.T , axis = 1, keepdims = True)), axis = 0, keepdims = True)

	# E Step
	def _expectation(self, X):
		numerator = self._probability(X) * self.alpha.T
		self.weights = numerator/np.sum(numerator, axis = 1, keepdims = True)
		self.weights = np.clip(self.weights, 0.001, 0.999)

	# M Step
	def _maximization(self, X):
		self.alpha = np.sum(self.weights, axis = 0, keepdims = True).T / X.shape[0]
		self.alpha = np.clip(self.alpha, 0.001, 0.999)

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
		
		p = (1/np.power(2*np.pi, self.dim/2)) * (1/np.sqrt(np.abs(np.linalg.det(self.covariance.T).reshape(1,self.n_comp)))) * np.exp(exp_term)
		if (self.clip == True):
			p = np.clip(p, 0.0000001, 100000000)
		
		return p

# Class Conditional Distribution is multinormal
class multinormal_model:
    
    def __init__(self, n_variables=1 ):
        self.n_variables = n_variables
        self.mean_vector = np.zeros((self.n_variables,))
        self.covariance = np.zeros((self.n_variables,self.n_variables))
        
        
    def estimate_parameters(self, X_train):
        '''ONly pass a 2d np array, if column vector reshape(n_rows,1)'''
        
        self.n_variables= (X_train).shape[1]
        mu = np.divide(X_train.sum(axis=0),(X_train).shape[0])
        self.mean_vector = mu
        sum_matrix = self.covariance
        for i in range(0,(X_train).shape[0]):
            X_i = X_train[i,:]
            a=np.tensordot((X_i-self.mean_vector),(X_i-self.mean_vector),axes=0)
            sum_matrix = np.add(a,sum_matrix)
        self.covariance= np.divide(sum_matrix, (X_train).shape[0])
        return [self.mean_vector, self.covariance]
        
        
    def get_likelihood(self, X_test):
        X_u = X_test - self.mean_vector
        #exp_term = -0.5*np.diagonal(np.matmul(np.matmul(X_u,self.covariance),X_u.T)
        #p = (1/np.power(2*np.pi, self.n_variables/2))* (1/np.sqrt(np.abs(np.linalg.det(self.covariance))))* np.exp(exp_term)
        exp_term = -0.5*np.diagonal(np.matmul(np.matmul(X_u,self.covariance),X_u.T))
        
        p = ((1/np.power(2*np.pi, self.n_variables/2))* (1/np.sqrt(np.abs(np.linalg.det(self.covariance)))))* np.exp(exp_term)
        #return np.array([p]).T                                    
        return np.array([p]).T

# Class conditional Distribution is multinomial
class multinomial_mle:
    '''This is for the special case for dicrete valued features in Naive Bayes '''
    #def __init__(self, n_vals):
    def __init__(self):
        
        #no of discrete values in the feature vector
        #self.n_vals = n_vals
        self.proba = {}
        
    def estimate_parameters(self, X_train):
        # X_train is a discrete valued column vector
        unique, counts = np.unique(X_train, return_counts=True)
        prob = counts/len(X_train)
        self.proba=dict(zip(unique, prob))
        # returns a dict with probability associated with each discrete element
        return self.proba
        
    
    def get_likelihood(self, X_test):
        p=[]
        for i in X_test:
            if i in self.proba:
                p.append(self.proba[i])
            else:
                p.append(0)
        # returns a mX1 ndarray with prob corresponding to each point        
        return np.array([p]).T

# Parzen Window
# Theory reference - http://www.cs.haifa.ac.il/~rita/ml_course/lectures_old/9_ParzenWin.pdf
class parzen_window:

	# Structure reference - http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.__init__
	def __init__(self, window_type = 'hypercube', h, dimension):
		self.window_type = window_type
		self.h = h
		self.dim = dimension
	
	def estimate_parameters(self, X_train):
		self.X = X_train	

	def _self.distance(X):
		if (window_type == 'hypercube'):
			return _hypercube_kernel_estimation(X)
		else if (window_type == 'gaussian'):
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
