import numpy as np

class PCA:
     
    # initialization
    def _init_(self, n_components):
        # number of components to be projected upon
        self.n_components = n_components
        # U matrix obtained after SVD of X
        self.U = []
        # mean of input data
        self.mean = []
        # S.D. of input data
        self.sigma = []

    # Calculate mean
    def _mean(self):
        [m, n] = np.shape(self.X)
        self.mean = (np.sum(self.X, axis = 0))/m

    # Calculate standard deviation
    def _sigma(self):
        variance = np.var(self.X, axis = 0)
        self.sigma = np.sqrt(variance)
        
    # Normalize data (Centering by subtracting mean)
    def _normalize(self):
        [m, n] = np.shape(self.X)
        self._mean()
        X_norm = (self.X - self.mean)
        return X_norm
    
    # Calculate covariance matrix
    def _covariance(self, X_norm):
        [m, n] = np.shape(X_norm)        
        return (np.matmul(np.transpose(X_norm), X_norm))/(m-1)
    
    # Singular Value Decomposition
    def _svd(self, X_norm):
        return np.linalg.svd(X_norm)
    
    # Projecting input data to smaller dimension
    def _projectData(self, X_norm):
        k = self.n_components
        return np.matmul(X_norm, self.U[:, 0:k])
        
    # Function to be called for calculating U and mean
    def fit(self, X_train, n_components):
        self.X = X_train
        self.n_components = n_components
        X_norm = self._normalize()
        [U, S, V] = self._svd(X_norm.T)
        self.U = U

    # Function to give projected data 
    def transform(self, X_test):
        X_norm = X_test - self.mean
        return self._projectData(X_norm)
