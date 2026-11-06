# weights_practice_v9.py
# PRACTICE PASS 9
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
# 6: Added neighborhood averaging before the second max pool,
#    with a constant channel for correct support padding.
#    AUROC .7403; accuracy .6758; sensitivity .6413; specificity .7639.
# 7: Unchanged neighborhood-averaging candidate.
#    AUROC .7291; accuracy .6719; sensitivity .6349; specificity .7761.
#    Discarded averaging: approximate matched AUROC on 242 shared
#    images declined from .7360 to .7255.
# 8: Restored pass 4/5 design, then narrowed the vertical component
#    of the common conv1 smoothing kernel.
#    AUROC .7929; accuracy .6367; sensitivity .5561; specificity .9000.
#    Approximate matched AUROC: .7898 original versus .7926 narrower.
#    Directional kernel remains under evaluation.
# 9: Identical weights to pass 8; extend directional-kernel evaluation.
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

    # Identical directional smoothing kernel to pass 8.
    # Both one-dimensional kernels are nonnegative and sum to one.
    horizontal = np.array(
        [1, 4, 6, 4, 1], dtype=np.float32
    ) / np.float32(16.0)

    vertical = np.array(
        [0, 1, 2, 1, 0], dtype=np.float32
    ) / np.float32(4.0)

    # Outer-product order is [kernel_row, kernel_col].
    smooth = np.outer(vertical, horizontal)

    thresholds = np.array(
        [1, 3, 10, 14, 18, 22, 26], dtype=np.float32
    ) / np.float32(32.0)

    # After the first ReLU and max pool:
    # h[i] = max(thresholds[i] - m, 0),
    # where m is the minimum directionally smoothed brightness
    # in that pooling window.
    for channel in range(7):
        conv1_weight[channel, 0] = -smooth
        conv1_bias[channel] = thresholds[channel]

    # Conv1 channels 7..15 remain identically zero.

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
    # Only the central conv2 spatial coefficient is used.
    for channel in range(4):
        conv2_weight[channel, channel + 3, 1, 1] = 8.0
        conv2_weight[channel, channel + 2, 1, 1] = -8.0
        conv2_weight[channel, 1, 1, 1] = -16.0
        conv2_weight[channel, 0, 1, 1] = 16.0

    # Pointwise nonblack support:
    # B = 1 - 16*(h[1] - h[0]).
    conv2_weight[4, 1, 1, 1] = -16.0
    conv2_weight[4, 0, 1, 1] = 16.0
    conv2_bias[4] = 1.0

    # Conv2 channels 5..31 remain identically zero.
    # The second max pool selects each indicator separately.

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
