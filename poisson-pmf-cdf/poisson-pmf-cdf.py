import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    values = np.arange(k+1)
    factorial = np.vectorize(math.factorial)
    pmf = np.exp(-lam)*lam**values / factorial(values)

    cdf = pmf.sum()
    return pmf[-1], cdf