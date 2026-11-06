# weights_practice_v7.py
# PRACTICE PASS 7
#
# Pass history:
# 1: Initial hand-designed dark-region baseline.
#    AUROC .7739; accuracy .3594; sensitivity .8947; specificity .2060.
# 2: Unchanged baseline on another batch.
#    AUROC .7559; accuracy .7344; sensitivity .8821; specificity .2623.
# 3: Changed fc.bias from log(3) to zero.
#    AUROC .7263; accuracy .6719; sensitivity .6264; specificity .7838.
#    Retained: matched accuracy unchanged, balanced accuracy improved.
# 4: Added quarter-strength spatial-reference correction.
#    AUROC .7388; accuracy .6797; sensitivity .6126; specificity .8769.
# 5: Unchanged reference-correction candidate.
#    AUROC .7913; accuracy .6484; sensitivity .5722; specificity .8871.
#    Retained correction: approximate matched AUROC on 242 shared
#    images improved from .756 to .787.
# 6: Added 3x3 averaging of existing indicators before second max pool,
#    with a constant channel for correct support padding.
#    AUROC .7403; accuracy .6758; sensitivity .6413; specificity .7639.
#    Averaging remains under evaluation.
# 7: Identical weights to pass 6; extend averaging evaluation.
#
# Constants are hand chosen, not fitted to per-image scores.

import numpy as np

def make_weights():
    """Return the six deterministic float32 parameter arrays."""

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=np.float32)
    conv1_bias = np.zeros((16,), dtype=np.float32)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=np.float32)
    conv2_bias = np.zeros((32,), dtype=np.float32)
    fc_weight = np.zeros((1, 512), dtype=np.float32)
    fc_bias = np.zeros((1,), dtype=np.float32)

    # Nonnegative, sum-one 5x5 smoothing kernel.
    binomial5 = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    smooth = np.outer(binomial5, binomial5) / np.float32(256.0)

    thresholds = np.array(
        [1, 3, 10, 14, 18, 22, 26], dtype=np.float32
    ) / np.float32(32.0)

    # After the first ReLU and max pool:
    # h[i] = max(thresholds[i] - m, 0),
    # where m is the minimum smoothed brightness in that window.
    for channel in range(7):
        conv1_weight[channel, 0] = -smooth
        conv1_bias[channel] = thresholds[channel]

    # Constant-one channel, preserved by ReLU and first max pool.
    # Conv2 padding makes it zero outside the valid feature map.
    conv1_bias[7] = 1.0

    # Conv1 channels 8..15 remain identically zero.

    # Neighborhood averaging kernel:
    # [[1, 2, 1],
    #  [2, 4, 2],
    #  [1, 2, 1]] / 16.
    binomial3 = np.array([1, 2, 1], dtype=np.float32)
    local_average = (
        np.outer(binomial3, binomial3) / np.float32(16.0)
    )

    # Pointwise dark-region indicators:
    # A[j] = 8*(h[j+3] - h[j+2]) - 16*(h[1] - h[0]).
    #
    # Common rising interval: [1/32, 3/32].
    # Falling intervals:
    # A[0]: [10/32, 14/32]
    # A[1]: [14/32, 18/32]
    # A[2]: [18/32, 22/32]
    # A[3]: [22/32, 26/32]
    #
    # Conv2 computes the neighborhood average of each indicator.
    for channel in range(4):
        conv2_weight[channel, channel + 3] = 8.0 * local_average
        conv2_weight[channel, channel + 2] = -8.0 * local_average
        conv2_weight[channel, 1] = -16.0 * local_average
        conv2_weight[channel, 0] = 16.0 * local_average

    # Pointwise nonblack support:
    # B = h[7] - 16*(h[1] - h[0]).
    #
    # Average B with zero contribution outside the valid feature map.
    # Fully black neighborhoods give zero.
    conv2_weight[4, 7] = local_average
    conv2_weight[4, 1] = -16.0 * local_average
    conv2_weight[4, 0] = 16.0 * local_average

    # All conv2 biases remain zero.
    # Conv2 channels 5..31 remain identically zero.
    # The second max pool selects each averaged channel separately.

    lung_weights = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ) / np.float32(3.5)

    reference_weights = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Explicit full 4x4 reference grid.
    reference_weights = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ) / np.float32(4.0)

    reference_strength = np.float32(0.25)
    spatial_weights = (
        lung_weights - reference_strength * reference_weights
    )

    dark_weights = (0.4, 0.3, 0.2, 0.1)
    feature_scale = 6.0
    support_coefficient = 0.45

    # E = 0.45*B - sum(dark_weights[j]*A[j]), after pooling.
    # logit = 6*(lung_average(E) - 0.25*reference_average(E)).
    for row in range(4):
        for col in range(4):
            spatial = float(spatial_weights[row, col])
            cell = row * 4 + col

            for channel, coefficient in enumerate(dark_weights):
                fc_weight[0, channel * 16 + cell] = (
                    -feature_scale * coefficient * spatial
                )

            fc_weight[0, 4 * 16 + cell] = (
                feature_scale * support_coefficient * spatial
            )

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }
