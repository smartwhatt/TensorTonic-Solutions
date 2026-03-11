import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X).astype(float)
    X_mean = np.mean(X, axis=0) 
    X -= X_mean
    n = X.shape[0]
    C = 1/(n-1) * (X.T @ X)

    eigenvalues, eigenvectors = np.linalg.eig(C)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    W = sorted_eigenvectors[:, :k]

    return X @ W