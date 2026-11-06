# weights_practice_v4.py
# PRACTICE PASS 4
#
# Pass history:
# 1: Initial hand-designed dark-region baseline.
#    AUROC 0.7739; accuracy 0.3594;
#    sensitivity 0.8947; specificity 0.2060.
#
# 2: Identical weights on another practice batch.
#    AUROC 0.7559; accuracy 0.7344;
#    sensitivity 0.8821; specificity 0.2623.
#
# 3: Changed only fc.bias from log(3) to zero.
#    AUROC 0.7263; accuracy 0.6719;
#    sensitivity 0.6264; specificity 0.7838.
#    Retain zero bias: the matched, predeclared comparison
#    preserves accuracy and improves balanced accuracy.
#
# 4: Candidate spatial-reference correction.
#    Add opposite-sign FC contributions in four reference cells,
#    with total reference strength one quarter of the lung term.
#    Filters, existing lung-cell weights, and bias are unchanged.
#
# All constants are hand chosen; none are fitted to image scores.

import numpy as np

def make_weights():
    """Return the six deterministic float32 parameter arrays."""

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=np.float32)
    conv1_bias = np.zeros((16,), dtype=np.float32)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=np.float32)
    conv2_bias = np.zeros((32,), dtype=np.float32)
    fc_weight = np.zeros((1, 512), dtype=np.float32)
    fc_bias = np.zeros((1,), dtype=np.float32)

    # Nonnegative, sum-one smoothing kernel.
    binomial = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    smooth = np.outer(binomial, binomial) / np.float32(256.0)

    thresholds = np.array(
        [1, 3, 10, 14, 18, 22, 26], dtype=np.float32
    ) / np.float32(32.0)

    # After the first ReLU and max pool:
    # h[i] = max(thresholds[i] - m, 0),
    # where m is the minimum smoothed brightness in that window.
    for channel in range(7):
        conv1_weight[channel, 0] = -smooth
        conv1_bias[channel] = thresholds[channel]

    # Conv1 channels 7..15 remain identically zero.

    # Four trapezoidal dark-region indicators:
    # A[j] = 8*(h[j+3] - h[j+2]) - 16*(h[1] - h[0]).
    #
    # Common rising interval: [1/32, 3/32].
    # Falling intervals:
    # A[0]: [10/32, 14/32]
    # A[1]: [14/32, 18/32]
    # A[2]: [18/32, 22/32]
    # A[3]: [22/32, 26/32]
    for channel in range(4):
        conv2_weight[channel, channel + 3, 1, 1] = 8.0
        conv2_weight[channel, channel + 2, 1, 1] = -8.0
        conv2_weight[channel, 1, 1, 1] = -16.0
        conv2_weight[channel, 0, 1, 1] = 16.0

    # Nonblack support:
    # B = 1 - 16*(h[1] - h[0]).
    conv2_weight[4, 1, 1, 1] = -16.0
    conv2_weight[4, 0, 1, 1] = 16.0
    conv2_bias[4] = 1.0

    # Conv2 channels 5..31 remain identically zero.
    # The second max pool selects each channel's response separately.

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
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ) / np.float32(4.0)

    # Single new feature-design change:
    # subtract a quarter-strength reference-region contribution.
    reference_strength = np.float32(0.25)
    spatial_weights = (
        lung_weights - reference_strength * reference_weights
    )

    dark_weights = (0.4, 0.3, 0.2, 0.1)
    feature_scale = 6.0
    support_coefficient = 0.45

    # Define E = 0.45*B - sum(dark_weights[j]*A[j]).
    # logit = 6 * (lung_average(E) - 0.25*reference_average(E)).
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
