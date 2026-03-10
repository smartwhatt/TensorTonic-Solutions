import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if len(y) == 0:
        return 0.0
    
    y = np.array(y)
    unique_values, counts = np.unique(y, return_counts=True)
    dist = counts/y.shape[0]
    return -sum(np.where(dist == 0, 0, dist*np.log2(dist)))