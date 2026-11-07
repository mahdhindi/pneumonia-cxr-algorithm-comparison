# edit_v2.py
import numpy as np

def edit(weight, bias):
    """Return 2,048 edited weights and one bias."""
    new_weight = np.asarray(weight, dtype=np.float64).copy()
    new_bias = float(bias)

    # Retain mean-centering.
    new_weight -= np.mean(new_weight)

    # Apply one fixed, weight-only clipping rule.
    rms = float(np.sqrt(np.mean(new_weight ** 2)))
    if rms > 0.0:
        limit = 1.5 * rms
        new_weight = np.clip(new_weight, -limit, limit)

    # Clipping can introduce a nonzero mean; remove it.
    new_weight -= np.mean(new_weight)

    return new_weight, new_bias
