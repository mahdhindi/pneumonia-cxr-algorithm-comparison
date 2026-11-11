# head_v14.py
import numpy as np

def make_head():
    """Return a 2,048-element weight array and one float bias."""
    weight = np.zeros(2048, dtype=np.float64)

    # Previously retained amplification.
    weight[1363] = 4.0
    weight[1766] = -2.0

    # f453 amplification remains under evaluation.
    weight[453] = 4.0

    # Remaining original pneumonia-associated features.
    weight[[1808, 101, 1653, 1401, 709, 2007]] = 1.0

    # Remaining original normal-associated features.
    weight[[423, 546, 1659]] = -1.0
    weight[[1582, 1327, 1751, 327]] = -0.25

    # Retained additions.
    weight[[868, 947, 948, 1346, 902]] = 0.5
    weight[[1132, 373, 1260, 1600, 2043]] = -0.5

    bias = 0.0
    return weight, bias