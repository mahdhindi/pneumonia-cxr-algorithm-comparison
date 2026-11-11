# weights_diag_v4.py
# PRACTICE PASS 4
#
# Pass 1: Hand-designed brightness and texture combination.
#         Reported AUROC: 0.5890.
# Pass 2: Uniform negative weighting of conv2 channel 8.
#         Reported AUROC: 0.9331.
# Pass 3: Change only fc.bias from +1.00 to +0.90.
#         Reported AUROC: 0.9177 on a different batch.
# Pass 4: Retain bias +0.90 and combine contrast at two scales:
#         channel 8:  -0.50 per cell
#         channel 26: -0.25 per cell
#         channel 27: -0.25 per cell
#         All convolutional parameters remain unchanged.
#
# No training, fitting, random numbers, or data-derived coefficients.

import numpy as np


def make_weights():
    """Return the six float32 parameter arrays for WrittenCNN."""
    dtype = np.float32

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=dtype)
    conv1_bias = np.zeros(16, dtype=dtype)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=dtype)
    conv2_bias = np.zeros(32, dtype=dtype)
    fc_weight = np.zeros((1, 512), dtype=dtype)
    fc_bias = np.array([0.90], dtype=dtype)

    # ----- Conv1: unchanged -----

    mean5 = np.full((5, 5), 1.0 / 25.0, dtype=dtype)

    center3 = np.zeros((5, 5), dtype=dtype)
    center3[1:4, 1:4] = 1.0 / 9.0

    ring5 = np.ones((5, 5), dtype=dtype)
    ring5[1:4, 1:4] = 0.0
    ring5 /= 16.0
    detail = center3 - ring5

    grad_x = np.zeros((5, 5), dtype=dtype)
    grad_x[:, :2] = -1.0 / 10.0
    grad_x[:, 3:] = 1.0 / 10.0
    grad_y = grad_x.T.copy()

    coordinates = np.arange(-2, 3, dtype=dtype)
    yy = coordinates[:, None]
    xx = coordinates[None, :]
    grad_diag = np.sign(xx + yy).astype(dtype) / 10.0
    grad_anti = np.sign(xx - yy).astype(dtype) / 10.0

    line_h = np.zeros((5, 5), dtype=dtype)
    line_h[2, :] = 1.0 / 5.0
    line_h[0, :] = -1.0 / 10.0
    line_h[4, :] = -1.0 / 10.0
    line_v = line_h.T.copy()

    conv1_weight[0, 0] = mean5
    conv1_weight[1, 0] = -mean5
    conv1_bias[1] = 1.0

    paired_filters = (
        detail,
        grad_x,
        grad_y,
        grad_diag,
        grad_anti,
        line_h,
        line_v,
    )
    for pair_index, kernel in enumerate(paired_filters):
        channel = 2 + 2 * pair_index
        conv1_weight[channel, 0] = kernel
        conv1_weight[channel + 1, 0] = -kernel

    # ----- Conv2: unchanged -----

    def coefficients(*entries):
        vector = np.zeros(16, dtype=dtype)
        for channel, value in entries:
            vector[channel] += value
        return vector

    # h0 = max(local mean).
    # h1 = 1 - min(local mean).
    # Hence h0 + h1 - 1 is the local range.
    m = coefficients((0, 0.5), (1, -0.5))
    local_range = coefficients((0, 1.0), (1, 1.0))
    t = coefficients((2, 1.0), (3, 1.0))
    gx = coefficients((4, 1.0), (5, 1.0))
    gy = coefficients((6, 1.0), (7, 1.0))
    gd = coefficients((8, 1.0), (9, 1.0))
    ga = coefficients((10, 1.0), (11, 1.0))
    horizontal = coefficients((12, 1.0), (13, 1.0))
    vertical = coefficients((14, 1.0), (15, 1.0))
    edge = (gx + gy + gd + ga) / 4.0

    mean3 = np.full((3, 3), 1.0 / 9.0, dtype=dtype)

    def put_conv2(channel, vector, bias=0.0, kernel=None):
        if kernel is None:
            kernel = mean3
        conv2_weight[channel] = (
            vector[:, None, None] * kernel[None, :, :]
        )
        conv2_bias[channel] = bias

    # Brightness and darkness.
    put_conv2(0, m, 0.50)
    put_conv2(1, -m, 0.50)

    for channel, threshold in (
        (2, 0.25),
        (3, 0.45),
        (4, 0.65),
        (5, 0.80),
    ):
        put_conv2(channel, m, 0.50 - threshold)

    put_conv2(6, -m, 0.65 - 0.50)
    put_conv2(7, -m, 0.45 - 0.50)

    # Local range, detail, gradients, and lines.
    put_conv2(8, local_range, -1.0)
    put_conv2(9, t)
    put_conv2(10, gx)
    put_conv2(11, gy)
    put_conv2(12, gd)
    put_conv2(13, ga)
    put_conv2(14, horizontal)
    put_conv2(15, vertical)

    # Bright/dark regions with an edge penalty.
    put_conv2(16, m - 0.5 * edge, 0.50 - 0.25)
    put_conv2(17, m - 0.5 * edge, 0.50 - 0.45)
    put_conv2(18, -m - 0.5 * edge, 0.65 - 0.50)
    put_conv2(19, -m - 0.5 * edge, 0.45 - 0.50)

    # Relative texture features.
    put_conv2(20, t - 0.5 * horizontal)
    put_conv2(21, vertical - 0.5 * horizontal)
    put_conv2(22, horizontal - 0.5 * vertical)
    put_conv2(23, 0.5 * (gd + ga) - 0.5 * gy)

    # Coarser spatial contrasts.
    surround3 = np.full((3, 3), -1.0 / 8.0, dtype=dtype)
    surround3[1, 1] = 1.0

    coarse_x = np.zeros((3, 3), dtype=dtype)
    coarse_x[:, 0] = -1.0 / 3.0
    coarse_x[:, 2] = 1.0 / 3.0
    coarse_y = coarse_x.T.copy()

    put_conv2(24, m, kernel=surround3)
    put_conv2(25, -m, kernel=surround3)
    put_conv2(26, m, kernel=coarse_x)
    put_conv2(27, -m, kernel=coarse_x)
    put_conv2(28, m, kernel=coarse_y)
    put_conv2(29, -m, kernel=coarse_y)

    # Detail combined with brightness or darkness.
    put_conv2(30, m + 2.0 * t, 0.0)
    put_conv2(31, -m + 2.0 * t, 0.0)

    # ----- Revised final layer -----
    # Equal weight across fine and coarse contrast measurements.
    # Split the coarse share equally between opposite polarities.
    # Apply each coefficient uniformly across all 16 grid cells.
    # Flattening order: channel * 16 + row * 4 + column.
    fc_weight[0, 8 * 16 : 9 * 16] = -0.50
    fc_weight[0, 26 * 16 : 27 * 16] = -0.25
    fc_weight[0, 27 * 16 : 28 * 16] = -0.25

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }