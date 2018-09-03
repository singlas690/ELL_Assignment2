import numpy as np

class PCA:
     
    def _init_(self, n_components = 3):
        self.n_components = n_components

    def _mean(self):
        [m, n] = np.shape(self.X)
        return (np.sum(self.X, axis = 0))/m
        
    def _sigma(self):
        variance = np.var(self.X, axis = 0)
        sigma = np.sqrt(variance)
        return sigma
        
    def _normalize(self):
        [m, n] = np.shape(self.X)
        mu = self._mean()
        sigma = self._sigma()
        X_norm = (self.X - mu)/sigma
        return X_norm
    
    def _covariance(self, X_norm):
        [m, n] = np.shape(X_norm)        
        return (np.matmul(np.transpose(X_norm), X_norm))/m
    
    def _svd(self, covariance):
        return np.linalg.svd(covariance)
    
    def _projectData(self, X_norm, U):
        k = self.n_components
        return np.matmul(X_norm, U[:, 1:k+1])
        
    def reduction(self, X_train, n_components):
        self.X = X_train
        self.n_components = n_components
        X_norm = self._normalize()
        covariance = self._covariance(X_norm)
        [U, S, V] = self._svd(covariance)
        X_project = self._projectData(X_norm, U)
        return X_project
