# head_v2.py
import numpy as np

def make_head():
    """Return a 2,048-element weight array and one float bias."""
    weight = np.zeros(2048, dtype=np.float64)

    # Pneumonia-associated features; unchanged from PASS 1.
    weight[[1363, 1808, 101, 453, 1653, 1401, 709, 2007]] = 1.0

    # Normal-associated features with smaller activations.
    weight[[1766, 423, 546, 1659]] = -1.0

    # Normal-associated features with larger activations.
    weight[[1582, 1327, 1751, 327]] = -0.25

    bias = 0.0
    return weight, bias