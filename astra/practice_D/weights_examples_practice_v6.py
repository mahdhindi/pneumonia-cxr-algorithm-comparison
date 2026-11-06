# weights_examples_practice_v6.py
# PRACTICE PASS 6
#
# History:
# Pass 1: original hand-designed network; AUROC 0.5474.
# Pass 2: removed fixed lung-versus-cardiac brightness comparison
#         and its +0.60 bias offset; AUROC 0.7097.
# Pass 3: changed only fc.bias from +0.10 to -0.15;
#         AUROC 0.6882.
# Pass 4: redistributed final spatial weights to central columns;
#         AUROC 0.7282.
# Pass 5: opacity channels use averaged local minimum brightness;
#         AUROC 0.7931. Current provisional baseline.
# Pass 6: replace triangular darkness response with a capped ramp.
#         Only final-layer channels 16 and 17 change.
#
# Practice passes use different batches.
import numpy as np

def make_weights():
    """Return six deterministic, manually designed float32 parameter arrays."""
    dtype = np.float32

    w1 = np.zeros((16, 1, 5, 5), dtype=dtype)
    b1 = np.zeros((16,), dtype=dtype)
    w2 = np.zeros((32, 16, 3, 3), dtype=dtype)
    b2 = np.zeros((32,), dtype=dtype)
    head = np.zeros((32, 4, 4), dtype=dtype)

    # Normalize fixed difference kernels using their coefficients only.
    def normalize_difference(kernel):
        kernel = np.asarray(kernel, dtype=dtype)
        return kernel / np.maximum(kernel, 0.0).sum()

    smooth = np.array([1, 4, 6, 4, 1], dtype=dtype) / 16.0
    derivative = np.array([-1, -2, 0, 2, 1], dtype=dtype)
    ridge = np.array([-1, -1, 4, -1, -1], dtype=dtype)
    narrow = np.array([0, 1, 2, 1, 0], dtype=dtype) / 4.0

    blur = np.outer(smooth, smooth)
    gx = normalize_difference(np.outer(smooth, derivative))
    gy = normalize_difference(np.outer(derivative, smooth))

    # Conv1 channels 0 and 1: brightness and darkness.
    w1[0, 0] = blur
    w1[1, 0] = -blur
    b1[1] = 1.0

    # Opposite-polarity pairs:
    # x, y, two diagonals, horizontal/vertical ridges, spots.
    difference_kernels = (
        gx,
        gy,
        normalize_difference(gx + gy),
        normalize_difference(gx - gy),
        normalize_difference(np.outer(ridge, smooth)),
        normalize_difference(np.outer(smooth, ridge)),
        normalize_difference(np.outer(narrow, narrow) - blur),
    )
    for index, kernel in enumerate(difference_kernels):
        channel = 2 + 2 * index
        w1[channel, 0] = kernel
        w1[channel + 1, 0] = -kernel

    average_axis = np.array([1, 2, 1], dtype=dtype) / 4.0
    average = np.outer(average_axis, average_axis)

    # Brightness midpoint M.
    # h0 is a local brightness maximum; h1 is 1 - local minimum.
    w2[0, 0] = 0.5 * average
    w2[0, 1] = -0.5 * average
    b2[0] = 0.5

    # Darkness D = 1 - M.
    w2[1] = -w2[0]
    b2[1] = 0.5

    # Conv2 channels 2..5: four orientation magnitudes.
    for output, first_input in enumerate((2, 4, 6, 8), start=2):
        w2[output, first_input] = average
        w2[output, first_input + 1] = average

    # Channels 6..9: ridge magnitudes and signed spot responses.
    w2[6, 10] = average
    w2[6, 11] = average
    w2[7, 12] = average
    w2[7, 13] = average
    w2[8, 14] = average
    w2[9, 15] = average

    # Channel 10: first-pool brightness range.
    w2[10, 0] = average
    w2[10, 1] = average
    b2[10] = -1.0

    # Channels 11 and 12: larger-scale midpoint center-surround.
    surround = np.full((3, 3), -1.0 / 8.0, dtype=dtype)
    surround[1, 1] = 1.0
    w2[11, 0] = 0.5 * surround
    w2[11, 1] = -0.5 * surround
    w2[12] = -w2[11]

    # F = 1 - average(h1): averaged local brightness minimum.
    #
    # S = F - 0.5*G - 0.25*(horizontal + vertical ridges)
    #       - 0.15*(bright + dark spots).
    #
    # The constant 1 in F is included in the biases below.
    smooth_bright = np.zeros((16, 3, 3), dtype=dtype)
    smooth_bright[1] = -average

    for channel in range(2, 10):
        smooth_bright[channel] -= 0.125 * average
    for channel in range(10, 14):
        smooth_bright[channel] -= 0.25 * average
    for channel in range(14, 16):
        smooth_bright[channel] -= 0.15 * average

    # O: capped opacity response.
    w2[13] = smooth_bright
    b2[13] = 1.0 - 0.30
    w2[14] = smooth_bright
    b2[14] = 1.0 - 0.65

    # Darkness threshold channels remain unchanged.
    # Channel 17 is retained but has zero final-layer weight in v6.
    for channel, threshold in zip((15, 16, 17), (0.55, 0.75, 0.95)):
        w2[channel] = w2[1]
        b2[channel] = 0.5 - threshold

    # Conv2 channels 18..31 remain zero.

    # Unchanged central spatial mask.
    lung = np.array(
        [
            [0.00, 0.0000, 0.0000, 0.00],
            [0.00, 0.2900, 0.2900, 0.00],
            [0.00, 0.3325, 0.0875, 0.00],
            [0.00, 0.0000, 0.0000, 0.00],
        ],
        dtype=dtype,
    )

    # Channels 0, 1, and 6..9 have zero direct final-layer weight.

    # -G: average the four orientation magnitudes.
    for channel in range(2, 6):
        head[channel] = -0.25 * lung

    # -0.5*R + 0.5*(B_positive - B_negative).
    head[10] = -0.5 * lung
    head[11] = 0.5 * lung
    head[12] = -0.5 * lung

    # +2*O.
    head[13] = 2.0 * lung
    head[14] = -2.0 * lung

    # Pass 6 change:
    # A = ReLU(D-.55) - ReLU(D-.75), capped at 0.20.
    # The final contribution is -2*A.
    head[15] = -2.0 * lung
    head[16] = 2.0 * lung
    # head[17] remains zero.

    # Unchanged final bias.
    bias = (
        -2.0 * 0.10
        + 2.0 * 0.10
        + 0.06
        + 0.5 * 0.08
        - 0.25
    )

    return {
        "conv1.weight": w1,
        "conv1.bias": b1,
        "conv2.weight": w2,
        "conv2.bias": b2,
        "fc.weight": head.reshape(1, 512),
        "fc.bias": np.array([bias], dtype=dtype),
    }
