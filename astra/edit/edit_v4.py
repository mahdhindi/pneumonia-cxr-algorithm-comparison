# edit_v4.py
# Apply once to the original, unedited layer.
import numpy as np
import math

def edit(weight, bias):
    """Return 2,048 edited weights and one bias."""
    new_weight = np.asarray(weight, dtype=np.float64).copy()

    # Reproduce Pass 3: center, prune, and recenter survivors.
    new_weight -= np.mean(new_weight)
    rms = float(np.sqrt(np.mean(new_weight ** 2)))

    if rms > 0.0:
        keep = np.abs(new_weight) >= 0.25 * rms
        new_weight[~keep] = 0.0
        new_weight[keep] -= np.mean(new_weight[keep])

    # Fixed odds-halving adjustment; no fitted threshold.
    new_bias = float(bias) - math.log(2.0)

    return new_weight, new_bias
