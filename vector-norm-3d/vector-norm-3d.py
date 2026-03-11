import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray(v)
    v = np.atleast_2d(v)
    result = np.linalg.norm(v, axis=1)
    return result[0] if result.shape[0] == 1 else result