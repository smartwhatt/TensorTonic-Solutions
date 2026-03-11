import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = lambda pos, j: np.where(j % 2 == 0, np.sin(pos/(10000**(j/d_model))), np.cos(pos/(10000**((j-1)/d_model))))
    
    return np.fromfunction(pe, (seq_length, d_model))