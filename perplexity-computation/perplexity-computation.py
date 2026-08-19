import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    prob_distributions = np.array(prob_distributions)
    actual_tokens = np.array(actual_tokens)
    p = np.array([prob_distributions[t][actual_tokens[t]] for t in range(len(actual_tokens))])

    H = -1/len(actual_tokens) * np.log(p).sum()
    return np.exp(H)