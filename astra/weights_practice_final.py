# weights_practice_final.py
# FINAL
#
# Final parameters reproduce the isotropic reference model
# evaluated in practice passes 4 and 5.
#
# Complete practice-pass history:
#
# 1: Initial hand-designed dark-region baseline.
#    Purpose: detect pooled dark regions in approximate lung cells,
#    suppress entirely black regions, and reduce cardiac contribution.
#    AUROC .7739.
#
# 2: Identical baseline weights.
#    Purpose: extend baseline evaluation across practice batches.
#    AUROC .7559.
#
# 3: Changed only fc.bias from log(3) to zero.
#    Purpose: remove the assumed development-prevalence offset.
#    Retained: matched accuracy was unchanged and balanced accuracy
#    improved, with a sensitivity/specificity tradeoff.
#    AUROC .7263.
#
# 4: Added quarter-strength spatial-reference correction.
#    Purpose: reduce some shared brightness/exposure effects using
#    upper and lower central reference cells.
#    AUROC .7388.
#
# 5: Identical reference-correction weights.
#    Purpose: evaluate the correction on additional practice images.
#    Retained: approximate matched AUROC on 242 shared images
#    improved from .756 to .787.
#    AUROC .7913.
#
# 6: Averaged the existing indicators over a 3x3 neighborhood
#    before the second max pool; added a constant channel for
#    correct support-indicator padding.
#    Purpose: favor dark regions supported by neighboring positions.
#    AUROC .7403.
#
# 7: Identical neighborhood-averaging weights.
#    Purpose: isolate averaging in a matched comparison.
#    Discarded: approximate matched AUROC declined .7360 to .7255.
#    AUROC .7291.
#
# 8: Restored the pointwise reference model, then narrowed only
#    the vertical component of conv1 smoothing.
#    Purpose: preserve thin dark gaps between roughly horizontal
#    rib patterns while retaining horizontal smoothing.
#    Approximate matched AUROC improved .7898 to .7926.
#    AUROC .7929.
#
# 9: Identical directional-filter weights.
#    Purpose: extend evaluation to another practice batch.
#    AUROC .7733.
#
# 10: Identical directional-filter weights.
#     Purpose: obtain another comparison before final selection.
#     Latest matched ranking was approximately tied near .72 AUROC;
#     threshold accuracy was 149/228 versus 151/228 for isotropic.
#     The small directional-filter benefit was not compelling enough
#     to retain; final weights restore the isotropic reference model.
#     AUROC .7102.
#
# Reported per-pass AUROCs concern different batches.
# Matched estimates were manually derived from rounded results.
# Constants below were hand chosen, not fitted to per-image scores.

import numpy as np

def make_weights():
    """Return exactly the six required float32 parameter arrays."""

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=np.float32)
    conv1_bias = np.zeros((16,), dtype=np.float32)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=np.float32)
    conv2_bias = np.zeros((32,), dtype=np.float32)
    fc_weight = np.zeros((1, 512), dtype=np.float32)
    fc_bias = np.zeros((1,), dtype=np.float32)

    # Original isotropic, nonnegative, sum-one smoothing kernel.
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
    # Zero below 1/32; rises to one by 3/32.
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
