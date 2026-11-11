# head_v3.py
import numpy as np

def make_head():
    """Return a 2,048-element weight array and one float bias."""
    weight = np.zeros(2048, dtype=np.float64)

    # Original pneumonia-associated features.
    weight[[1363, 1808, 101, 453, 1653, 1401, 709, 2007]] = 1.0

    # Original normal-associated features.
    weight[[1766, 423, 546, 1659]] = -1.0
    weight[[1582, 1327, 1751, 327]] = -0.25

    # Additional pneumonia-associated features supported by both batches.
    weight[[868, 947, 948, 1346]] = 0.5

    # Additional normal-associated features supported by both batches.
    weight[[1132, 373, 1260, 1600]] = -0.5

    bias = 0.0
    return weight, bias