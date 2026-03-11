import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.atleast_2d(v)
    norms = np.linalg.norm(v, axis=1)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    result = (v.T / norms).T
    return result[0] if result.shape[0] == 1 else result