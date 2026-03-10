import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    
    y = np.array(y)
    _, counts = np.unique(y, return_counts=True)
    
    p = counts / counts.sum()
    p = p[p > 0]  # remove zero probabilities
    
    return -np.sum(p * np.log2(p))