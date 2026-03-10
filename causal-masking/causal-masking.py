import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    mask = np.triu(np.ones_like(scores), k=1) > 0
    new_scores = scores.copy()
    new_scores[mask] = mask_value

    return new_scores

    