import numpy as np
def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    # Write code here
    c = size // 2
    i = np.arange(size)[None, :]
    j = np.arange(size)[:, None]
    x = j - c
    y = i - c

    result = np.exp(-((x**2 + y**2)/(2*sigma**2)))
    result /= result.sum()

    return result.tolist()