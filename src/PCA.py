import numpy as np

class PCA:
     
    # initialization
    def _init_(self, n_components):
        # number of components to be projected upon
        self.n_components = n_components

    # Calculate mean
    def _mean(self):
        [m, n] = np.shape(self.X)
        return (np.sum(self.X, axis = 0))/m

    # Calculate standard deviation
    def _sigma(self):
        variance = np.var(self.X, axis = 0)
        sigma = np.sqrt(variance)
        return sigma

    # Normalize data (Centering by subtracting mean)
    def _normalize(self):
        [m, n] = np.shape(self.X)
        mu = self._mean()
        X_norm = (self.X - mu)
        return X_norm
    
    # Calculate covariance matrix
    def _covariance(self, X_norm):
        [m, n] = np.shape(X_norm)        
        return (np.matmul(np.transpose(X_norm), X_norm))/(m-1)
    
    # Singular Value Decomposition
    def _svd(self, X_norm):
        return np.linalg.svd(X_norm)
    
    # Projecting input data to smaller dimension
    def _projectData(self, X_norm, U):
        k = self.n_components
        return np.matmul(X_norm, U[:, 1:k+1])
        
    # Function to be called for dimensionality reduction
    def reduction(self, X_train, n_components):
        self.X = X_train
        self.n_components = n_components
        X_norm = self._normalize()
        [U, S, V] = self._svd(X_norm)
        X_project = self._projectData(X_norm, U)
        return X_project
