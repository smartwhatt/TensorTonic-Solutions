import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = lambda pos, j: np.where(j % 2 == 0, np.sin(pos/(base**(j/d_model))), np.cos(pos/(base**((j-1)/d_model))))
    
    return np.fromfunction(pe, (seq_len, d_model))