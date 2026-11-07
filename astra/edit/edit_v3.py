# edit_v3.py
# Apply once to the original, unedited layer.
import numpy as np

def edit(weight, bias):
    """Return 2,048 edited weights and one bias."""
    new_weight = np.asarray(weight, dtype=np.float64).copy()
    new_bias = float(bias)

    # Reproduce the retained mean-centering edit.
    new_weight -= np.mean(new_weight)

    # Remove small centered coefficients using a fixed rule.
    rms = float(np.sqrt(np.mean(new_weight ** 2)))
    if rms > 0.0:
        keep = np.abs(new_weight) >= 0.25 * rms
        new_weight[~keep] = 0.0

        # Preserve zero total weight while keeping pruned entries zero.
        new_weight[keep] -= np.mean(new_weight[keep])

    return new_weight, new_bias
