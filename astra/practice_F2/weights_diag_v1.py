# weights_diag_v1.py
# PRACTICE PASS 1
# Fixed, hand-designed filters. No training, fitting, or randomness.

import numpy as np


def make_weights():
    """Return the six float32 parameter arrays for WrittenCNN."""
    dtype = np.float32

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=dtype)
    conv1_bias = np.zeros(16, dtype=dtype)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=dtype)
    conv2_bias = np.zeros(32, dtype=dtype)
    fc_weight = np.zeros((1, 512), dtype=dtype)
    fc_bias = np.array([-0.50], dtype=dtype)

    # ----- Conv1: brightness, detail, gradients, and lines -----

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

    # ----- Conv2 coefficient vectors -----

    def coefficients(*entries):
        vector = np.zeros(16, dtype=dtype)
        for channel, value in entries:
            vector[channel] += value
        return vector

    # Before spatial averaging:
    # h0 = max(local mean), h1 = 1 - min(local mean).
    # Thus 0.5*h0 - 0.5*h1 + 0.5 is their midpoint.
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

    # Texture and orientation measurements.
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

    # Coarser spatial contrasts. These kernels sum to zero.
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

    # Brightness/detail conjunction probes:
    # M + 2*T - 0.50 and (1-M) + 2*T - 0.50.
    put_conv2(30, m + 2.0 * t, 0.0)
    put_conv2(31, -m + 2.0 * t, 0.0)

    # ----- Final layer: fixed anatomical emphasis -----

    spatial_mask = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.5, 1.0],
            [1.0, 2.0, 0.75, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    ) / 10.25

    gains = np.zeros(32, dtype=dtype)
    gains[0] = 1.5
    gains[1] = -1.5
    gains[2] = 2.0
    gains[4] = -2.0
    gains[7] = -3.0
    gains[16] = 2.0
    gains[17] = -2.0
    gains[20] = 6.0
    gains[22] = -3.0

    # Remaining channels are diagnostic probes with zero contribution.
    # Flattening order: channel * 16 + row * 4 + column.
    fc_weight[0] = (
        gains[:, None, None] * spatial_mask[None, :, :]
    ).reshape(512)

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }