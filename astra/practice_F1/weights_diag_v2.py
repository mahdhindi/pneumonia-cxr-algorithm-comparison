# weights_diag_v2.py
# PRACTICE PASS 2
#
# Conv1 and conv2 are identical to practice pass 1.
# Only the FC weights and FC bias change.
# All values are manually specified constants or fixed formulas.
# No data access, fitting, randomness, or inference is performed.

import numpy as np

def make_weights():
    """Return the six float32 parameter arrays for WrittenCNN."""
    dtype = np.float32

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=dtype)
    conv1_bias = np.zeros((16,), dtype=dtype)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=dtype)
    conv2_bias = np.zeros((32,), dtype=dtype)
    fc_weight = np.zeros((1, 512), dtype=dtype)
    fc_bias = np.array([1.0], dtype=dtype)

    # ---------------------------------------------------------------
    # Conv1: unchanged from pass 1.
    # ---------------------------------------------------------------

    box5 = np.full((5, 5), 1.0 / 25.0, dtype=dtype)

    center3 = np.zeros((5, 5), dtype=dtype)
    center3[1:4, 1:4] = 1.0 / 9.0

    # Right-half mean minus left-half mean.
    grad_x = np.zeros((5, 5), dtype=dtype)
    grad_x[:, 0:2] = -1.0 / 10.0
    grad_x[:, 3:5] = 1.0 / 10.0
    grad_y = grad_x.T.copy()

    # Central row mean minus the mean of the two outer rows.
    horizontal_line = np.zeros((5, 5), dtype=dtype)
    horizontal_line[2, :] = 1.0 / 5.0
    horizontal_line[0, :] = -1.0 / 10.0
    horizontal_line[4, :] = -1.0 / 10.0
    vertical_line = horizontal_line.T.copy()

    diagonal_line = np.eye(5, dtype=dtype) / 5.0 - box5
    antidiagonal_line = np.fliplr(np.eye(5, dtype=dtype)) / 5.0 - box5

    # 0: local brightness.
    conv1_weight[0, 0] = box5

    # 1: darkness below 0.50.
    conv1_weight[1, 0] = -box5
    conv1_bias[1] = 0.50

    # 2: brightness above 0.35.
    conv1_weight[2, 0] = box5
    conv1_bias[2] = -0.35

    # 3: brightness above 0.60.
    conv1_weight[3, 0] = box5
    conv1_bias[3] = -0.60

    # 4 / 5: positive / negative fine center-surround contrast.
    conv1_weight[4, 0] = center3 - box5
    conv1_weight[5, 0] = box5 - center3

    # 6 / 7: horizontal gradient polarities.
    conv1_weight[6, 0] = grad_x
    conv1_weight[7, 0] = -grad_x

    # 8 / 9: vertical gradient polarities.
    conv1_weight[8, 0] = grad_y
    conv1_weight[9, 0] = -grad_y

    # 10 / 11: horizontal bright / dark lines.
    conv1_weight[10, 0] = horizontal_line
    conv1_weight[11, 0] = -horizontal_line

    # 12 / 13: vertical bright / dark lines.
    conv1_weight[12, 0] = vertical_line
    conv1_weight[13, 0] = -vertical_line

    # 14 / 15: diagonal / antidiagonal bright lines.
    conv1_weight[14, 0] = diagonal_line
    conv1_weight[15, 0] = antidiagonal_line

    # ---------------------------------------------------------------
    # Conv2: unchanged from pass 1.
    # ---------------------------------------------------------------

    avg3 = np.full((3, 3), 1.0 / 9.0, dtype=dtype)

    center = np.zeros((3, 3), dtype=dtype)
    center[1, 1] = 1.0

    ring = np.full((3, 3), 1.0 / 8.0, dtype=dtype)
    ring[1, 1] = 0.0
    broad_spot = center - ring

    broad_vertical = np.array(
        [
            [-0.5, 1.0, -0.5],
            [-0.5, 1.0, -0.5],
            [-0.5, 1.0, -0.5],
        ],
        dtype=dtype,
    ) / 3.0
    broad_horizontal = broad_vertical.T.copy()

    def set_channel(out_channel, terms, bias=0.0):
        """Assign one manually specified linear combination."""
        for in_channel, kernel in terms:
            conv2_weight[out_channel, in_channel] += kernel
        conv2_bias[out_channel] = bias

    # 0-3: smoothed conv1 brightness and darkness features.
    for channel in range(4):
        set_channel(channel, [(channel, avg3)])

    # 4 / 5: regional brightness above 0.25 / 0.45.
    set_channel(4, [(0, avg3)], bias=-0.25)
    set_channel(5, [(0, avg3)], bias=-0.45)

    # 6 / 7: regional darkness below 0.65 / 0.30.
    set_channel(6, [(0, -avg3)], bias=0.65)
    set_channel(7, [(0, -avg3)], bias=0.30)

    # 8 / 9: broad bright / dark spots.
    set_channel(8, [(0, broad_spot)], bias=-0.015)
    set_channel(9, [(0, -broad_spot)], bias=-0.015)

    # 10 / 11: broad vertical bright / dark bands.
    set_channel(10, [(0, broad_vertical)], bias=-0.01)
    set_channel(11, [(0, -broad_vertical)], bias=-0.01)

    # 12 / 13: broad horizontal bright / dark bands.
    set_channel(12, [(0, broad_horizontal)], bias=-0.01)
    set_channel(13, [(0, -broad_horizontal)], bias=-0.01)

    # 14: both polarities of fine contrast.
    set_channel(14, [(4, avg3), (5, avg3)])

    # 15 / 16: both polarities of horizontal / vertical gradients.
    set_channel(15, [(6, avg3), (7, avg3)])
    set_channel(16, [(8, avg3), (9, avg3)])

    # 17 / 18: horizontal bright / dark lines.
    set_channel(17, [(10, avg3)])
    set_channel(18, [(11, avg3)])

    # 19 / 20: vertical bright / dark lines.
    set_channel(19, [(12, avg3)])
    set_channel(20, [(13, avg3)])

    # 21: combined diagonal bright-line responses.
    set_channel(21, [(14, avg3), (15, avg3)])

    detail_penalty = [
        (4, -0.75 * avg3),
        (5, -0.75 * avg3),
        (6, -0.20 * avg3),
        (7, -0.20 * avg3),
        (8, -0.20 * avg3),
        (9, -0.20 * avg3),
    ]

    # 22: moderate brightness with sharp detail suppressed.
    set_channel(22, [(2, avg3)] + detail_penalty)

    # 23: high brightness with sharp detail suppressed.
    set_channel(23, [(3, avg3)] + detail_penalty)

    # 24: broad bright spots with narrow bright lines suppressed.
    set_channel(
        24,
        [
            (0, broad_spot),
            (10, -0.50 * avg3),
            (12, -0.25 * avg3),
        ],
        bias=-0.01,
    )

    # 25: broad dark spots with narrow dark lines suppressed.
    set_channel(
        25,
        [
            (0, -broad_spot),
            (11, -0.50 * avg3),
            (13, -0.25 * avg3),
        ],
        bias=-0.01,
    )

    # 26: dark regions with sharp detail suppressed.
    set_channel(
        26,
        [
            (1, avg3),
            (4, -0.50 * avg3),
            (5, -0.50 * avg3),
            (6, -0.10 * avg3),
            (7, -0.10 * avg3),
            (8, -0.10 * avg3),
            (9, -0.10 * avg3),
        ],
    )

    # 27: fine contrast with axial bright lines suppressed.
    set_channel(
        27,
        [
            (4, avg3),
            (5, avg3),
            (10, -0.40 * avg3),
            (12, -0.40 * avg3),
        ],
    )

    # 28: diagonal detail with axial bright lines suppressed.
    set_channel(
        28,
        [
            (14, avg3),
            (15, avg3),
            (10, -0.25 * avg3),
            (12, -0.25 * avg3),
        ],
    )

    # 29: predominantly horizontal line detail.
    set_channel(
        29,
        [
            (10, avg3),
            (11, avg3),
            (12, -0.50 * avg3),
            (13, -0.50 * avg3),
        ],
    )

    # 30: predominantly vertical line detail.
    set_channel(
        30,
        [
            (12, avg3),
            (13, avg3),
            (10, -0.50 * avg3),
            (11, -0.50 * avg3),
        ],
    )

    # 31: regional brightness above 0.75.
    set_channel(31, [(0, avg3)], bias=-0.75)

    # ---------------------------------------------------------------
    # FC: revised in pass 2. Spatial mask unchanged.
    # ---------------------------------------------------------------

    lung_mask = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.04, 0.26, 0.22, 0.04],
            [0.04, 0.22, 0.14, 0.04],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=dtype,
    )

    channel_coefficients = {
        14: -8.0,
        15: -2.0,
        16: -2.0,
        17: -2.0,
        21: -4.0,
        24: -2.0,
        27: -4.0,
    }

    for channel, coefficient in channel_coefficients.items():
        start = channel * 16
        fc_weight[0, start:start + 16] = (
            coefficient * lung_mask
        ).reshape(16)

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }
