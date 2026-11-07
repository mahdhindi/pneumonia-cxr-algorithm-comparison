# edit_v1.py
import numpy as np

def edit(weight, bias):
    """Return 2,048 edited weights and one bias."""
    new_weight = weight.copy()
    new_bias = bias

    # Remove the uniform component using only the supplied weights.
    mean_weight = float(np.mean(new_weight, dtype=np.float64))
    new_weight = new_weight - mean_weight

    return new_weight, new_bias
