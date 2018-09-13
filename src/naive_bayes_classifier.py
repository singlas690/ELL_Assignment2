import numpy as np 
import class_conditional_estimator as cc 

class naive_bayes_classifier:

	# Initialization
	def __init__(self, param, distrib):
		self.X = None
		self.Y = None

		# Parameters corresponding to the class conditional estimator
		self.parameters = param
		# List of class conditional distributions for each feature
		self.distributions = distrib
		# List of class conditional estimator model objects per class
		self.per_class_estimator = []
		# List of class priors
		self.prior = []

	# Input m examples each of dimension d with corresponding labels
	# X - [m x dim], Y - [m x 1]
	def train(self, X_train, Y_train):
		self.X = X_train
		self.Y = Y_train

		self._estimate_px_theta()

	# Returns a model object for the class conditional estimator
	def _get_est_object(self, feature_number):
		if self.distributions[feature_number] == 'GMM':
			comp = self.parameters[feature_number][0]
			toler = self.parameters[feature_number][1]
			max_it = self.parameters[feature_number][2]
			dim = self.parameters[feature_number][3]
			reg = self.parameters[feature_number][4]
			return cc.gaussian_mixture_model(n_comp = comp, tol = toler, max_iter = max_it, dimension = dim, regularize = reg)

		# Implement other estimators by Harsh
		if self.distributions[feature_number] == '':
			return None

		if self.distributions[feature_number] == '':
			return None

		if self.distributions[feature_number] == '':
			return None

	# Estimate parameters for class conditional estimators	
	def _estimate_px_theta(self):
		classes = np.unique(self.Y)
		for i in classes:
			X_temp = (self.X)[np.repeat(self.Y == i, (self.X).shape[1], axis = 1)]
			X_temp = X_temp.reshape(int(X_temp.shape[0] / (self.X).shape[1]), (self.X).shape[1] )

			# Calculate model parameters for each feature vector
			for j in range((self.X).shape[1]):
				model = self._get_est_object(j)
				# print(X_temp[:, j])
				model.estimate_parameters(np.expand_dims(X_temp[:, j], axis = 1))
				(self.per_class_estimator).append(model)

			(self.prior).append(np.sum(self.Y == i) / self.Y.shape[0])

		# 2D Matrix of weights for each class and feature distribution
		self.per_class_estimator = np.asarray(self.per_class_estimator)
		self.per_class_estimator = (self.per_class_estimator).reshape((classes.shape[0], (self.X).shape[1]))


	# Returns class label for given m test examples each of dimension dim, X_test = [m x dim]
	def predict(self, X_test):
		classes = np.expand_dims(np.unique(self.Y), axis = 1)
		Q = np.zeros((X_test.shape[0], classes.shape[0]))

		# Number of features
		n = X_test.shape[1]
		for i in range(classes.shape[0]):
			for j in range(n):
				# Calculate log of probability so that these can be added later since multiplication of probabilities will reduce to very small value
				Q[:, i] = Q[:, i] + np.log(((self.per_class_estimator)[i][j].get_likelihood(X_test[:, j]) * (self.prior)[i]).ravel())
		class_idx = np.argmax(Q, axis = 1)

		return classes[class_idx]


if __name__ == "__main__":
	num_ex = 100
	dim = 5
	X_train = (1000*np.random.rand(num_ex, dim)).astype('int')
	Y_train = (5*np.random.rand(num_ex, 1)).astype('int')

	parameters = [[1, 0.001, 20, 1, 0.0001]]*5
	distributions = ['GMM', 'GMM', 'GMM', 'GMM', 'GMM']

	model1 = naive_bayes_classifier(param = parameters, distrib = distributions)
	model1.train(X_train, Y_train)

	print("Train complete")
	X_test = (10*np.random.rand(10, dim)).astype('int')
	print(model1.predict(X_test))
