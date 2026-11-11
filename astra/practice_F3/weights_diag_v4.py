# weights_diag_v5.py
# PRACTICE PASS 5
#
# The supplied runner log labels the previous evaluation pass 4/20.
#
# Change from weights_diag_v3.py:
# Channel 12's FC spatial weights now emphasize the middle rows.
# Conv1, conv2, all other FC weights, and FC bias remain unchanged.
#
# No data access, randomness, fitting, or optimisation is used.

import numpy as np
import math


def make_weights():
    """Return the six float32 parameter arrays for WrittenCNN."""
    dtype = np.float32

    # ---------------------------------------------------------------
    # Conv1: unchanged.
    # ---------------------------------------------------------------
    conv1_weight = np.zeros((16, 1, 5, 5), dtype=dtype)
    conv1_bias = np.zeros(16, dtype=dtype)

    smooth5 = np.array([1, 4, 6, 4, 1], dtype=dtype) / 16.0
    blur5 = np.outer(smooth5, smooth5)

    smooth3 = np.array([1, 2, 1], dtype=dtype) / 4.0
    blur3_embedded = np.zeros((5, 5), dtype=dtype)
    blur3_embedded[1:4, 1:4] = np.outer(smooth3, smooth3)

    derivative = np.array([-1, -2, 0, 2, 1], dtype=dtype) / 3.0
    dx = np.outer(smooth5, derivative)
    dy = np.outer(derivative, smooth5)
    diagonal_a = (dx + dy) / math.sqrt(2.0)
    diagonal_b = (dx - dy) / math.sqrt(2.0)

    fine_detail = 2.0 * (blur3_embedded - blur5)
    ridge_rows = np.array([-1, -1, 4, -1, -1], dtype=dtype) / 4.0
    horizontal_ridge = np.outer(ridge_rows, smooth5)

    conv1_weight[0, 0] = blur5

    conv1_weight[1, 0] = -blur5
    conv1_bias[1] = 1.0

    conv1_weight[2, 0] = blur5
    conv1_bias[2] = -0.30

    conv1_weight[3, 0] = blur5
    conv1_bias[3] = -0.55

    conv1_weight[4, 0] = blur5
    conv1_bias[4] = -0.80

    conv1_weight[5, 0] = fine_detail
    conv1_weight[6, 0] = -fine_detail

    conv1_weight[7, 0] = dx
    conv1_weight[8, 0] = -dx
    conv1_weight[9, 0] = dy
    conv1_weight[10, 0] = -dy
    conv1_weight[11, 0] = diagonal_a
    conv1_weight[12, 0] = -diagonal_a
    conv1_weight[13, 0] = diagonal_b
    conv1_weight[14, 0] = -diagonal_b

    conv1_weight[15, 0] = horizontal_ridge

    # ---------------------------------------------------------------
    # Conv2: unchanged.
    # Each intermediate is a fixed [16, 3, 3] weight stencil.
    # ---------------------------------------------------------------
    average3 = np.outer(smooth3, smooth3)

    def smoothed(channel):
        stencil = np.zeros((16, 3, 3), dtype=dtype)
        stencil[channel] = average3
        return stencil

    brightness = smoothed(0)
    darkness = smoothed(1)
    above_030 = smoothed(2)
    above_055 = smoothed(3)
    above_080 = smoothed(4)

    midtone = 4.0 * (above_030 - 2.0 * above_055 + above_080)

    bright_detail = smoothed(5)
    dark_detail = smoothed(6)
    fine_magnitude = 0.5 * (bright_detail + dark_detail)

    edge_x = smoothed(7) + smoothed(8)
    edge_y = smoothed(9) + smoothed(10)
    edge_a = smoothed(11) + smoothed(12)
    edge_b = smoothed(13) + smoothed(14)
    edges = 0.25 * (edge_x + edge_y + edge_a + edge_b)
    ribs = smoothed(15)

    ring = np.array(
        [[1, 2, 1],
         [2, 0, 2],
         [1, 2, 1]],
        dtype=dtype,
    ) / 12.0

    center_surround = -ring
    center_surround[1, 1] = 1.0

    coarse_bump = np.zeros((16, 3, 3), dtype=dtype)
    coarse_bump[0] = center_surround

    vertical_band = np.zeros((16, 3, 3), dtype=dtype)
    vertical_band[0] = np.array(
        [[0.0, 0.0, 0.0],
         [-0.5, 1.0, -0.5],
         [0.0, 0.0, 0.0]],
        dtype=dtype,
    )

    horizontal_band = np.zeros((16, 3, 3), dtype=dtype)
    horizontal_band[0] = np.array(
        [[0.0, -0.5, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, -0.5, 0.0]],
        dtype=dtype,
    )

    conv2_weight = np.stack(
        [
            brightness,                              # 0
            darkness,                                # 1
            above_030,                               # 2
            above_055,                               # 3
            above_080,                               # 4
            midtone,                                 # 5
            bright_detail,                           # 6
            dark_detail,                             # 7
            edge_x,                                  # 8
            edge_y,                                  # 9
            edge_a,                                  # 10
            edge_b,                                  # 11
            edges,                                   # 12
            ribs,                                    # 13
            coarse_bump,                             # 14
            -coarse_bump,                            # 15
            vertical_band,                           # 16
            horizontal_band,                         # 17
            edge_y - edge_x,                         # 18
            edge_x - edge_y,                         # 19
            edge_a - edge_b,                         # 20
            edge_b - edge_a,                         # 21
            brightness - 1.5 * edges,                # 22
            brightness - 1.5 * edges,                # 23
            midtone - 2.0 * edges,                   # 24
            darkness - edges,                       # 25
            edges + 0.20 * brightness,               # 26
            fine_magnitude + 0.20 * brightness,      # 27
            coarse_bump + 0.20 * brightness,         # 28
            -coarse_bump + 0.20 * brightness,        # 29
            above_055 - 0.75 * ribs,                 # 30
            brightness + darkness + edges,          # 31
        ],
        axis=0,
    ).astype(dtype)

    conv2_bias = np.zeros(32, dtype=dtype)
    conv2_bias[22] = -0.35
    conv2_bias[23] = -0.55
    conv2_bias[25] = -0.55
    conv2_bias[26] = -0.13
    conv2_bias[27] = -0.12
    conv2_bias[28] = -0.12
    conv2_bias[29] = -0.12
    conv2_bias[31] = -1.06

    # ---------------------------------------------------------------
    # Revised spatial weighting for channel 12 only.
    # Absolute weights sum to 16, as in the uniform-weight candidate.
    # ---------------------------------------------------------------
    spatial = np.array(
        [[0.5, 0.5, 0.5, 0.5],
         [1.5, 1.5, 1.5, 1.5],
         [1.5, 1.5, 1.5, 1.5],
         [0.5, 0.5, 0.5, 0.5]],
        dtype=dtype,
    )

    fc_weight = np.zeros((1, 512), dtype=dtype)
    fc_weight[0, 12 * 16:13 * 16] = -spatial.reshape(16)

    fc_bias = np.array([1.25], dtype=dtype)

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }