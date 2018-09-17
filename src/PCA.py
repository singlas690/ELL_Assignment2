import numpy as np

class PCA:
     
    # initialization
    def __init__(self, n_components):
        # number of components to be projected upon
        self.n_components = n_components
        # U matrix obtained after SVD of X
        self.Qvec = []
        # mean of input data
        self.mean = []
        # S.D. of input data
        self.sigma = []
        self.explained_variance_ratio_ = 0

    # Normalize data (Centering by subtracting mean)
    def _normalize(self, X):
        self.mean = np.mean(X, axis = 0)
        X_norm = (X - self.mean)
        return X_norm

    # Function to be called for calculating U and mean
    def fit(self, X_train):
        X_norm = self._normalize(X_train)
        covar = np.dot(X_norm.T, X_norm)
        eival, eivec = np.linalg.eig(covar)

        idx_comp = (np.argsort(np.expand_dims(-eival, axis = 1), axis = 0))[:self.n_components, :]
        self.Qvec = (eivec[:, idx_comp])[:,:,0]

        sum_var = np.sum(np.var(X_norm, axis = 0))
        self.explained_variance_ratio_ = np.var(np.dot(X_norm, self.Qvec), axis = 0)/sum_var

    # Function to give projected data 
    def transform(self, X_test):
        X_test = X_test.astype('float')
        X_norm = self._normalize(X_test)
        return np.dot(X_norm, self.Qvec)
