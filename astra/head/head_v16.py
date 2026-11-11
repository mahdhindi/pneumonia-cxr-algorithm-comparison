# head_v16.py
import numpy as np

def make_head():
    """Return a 2,048-element weight array and one float bias."""
    weight = np.zeros(2048, dtype=np.float64)

    # Retained amplification.
    weight[1363] = 4.0
    weight[1766] = -2.0

    # Original pneumonia-associated features; f453 remains restored.
    weight[[1808, 101, 453, 1653, 1401, 709, 2007]] = 1.0

    # Original normal-associated features.
    weight[[423, 546, 1659]] = -1.0
    weight[[1582, 1327, 1751, 327]] = -0.25

    # Retained additions.
    weight[[868, 947,948, 1346, 902]] = 0.5
    weight[[1132, 373, 1260, 1600, 2043]] = -0.5

    # Unchanged positive offset under evaluation.
    bias = 0.25
    return weight, bias