# weights_examples_practice_v2.py
# PRACTICE PASS 2
#
# Pass 1: original hand-designed network; reported AUROC 0.5474.
# Pass 2: remove the fixed lung-versus-cardiac brightness comparison.
#         Set final weights for channels 0 and 1 to zero.
#         Remove its associated +0.60 bias offset.
#         All other parameter values are unchanged.
import numpy as np

def make_weights():
    """Return six deterministic, manually designed float32 parameter arrays."""
    dtype = np.float32

    w1 = np.zeros((16, 1, 5, 5), dtype=dtype)
    b1 = np.zeros((16,), dtype=dtype)
    w2 = np.zeros((32, 16, 3, 3), dtype=dtype)
    b2 = np.zeros((32,), dtype=dtype)
    head = np.zeros((32, 4, 4), dtype=dtype)

    # Fixed kernels. Normalization uses kernel coefficients only.
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

    # Channels 0 and 1: brightness and darkness.
    w1[0, 0] = blur
    w1[1, 0] = -blur
    b1[1] = 1.0

    # Seven predetermined opposite-polarity pairs.
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

    # Fixed averaging kernel on the first pooled feature maps.
    average_axis = np.array([1, 2, 1], dtype=dtype) / 4.0
    average = np.outer(average_axis, average_axis)

    # h0 is a local brightness maximum; h1 is 1 - local minimum.
    # M = 0.5 * average(h0 - h1) + 0.5.
    w2[0, 0] = 0.5 * average
    w2[0, 1] = -0.5 * average
    b2[0] = 0.5

    # D = 1 - M.
    w2[1] = -w2[0]
    b2[1] = 0.5

    # Channels 2..5: four orientation magnitudes.
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

    # Channel 10: average first-pool brightness range.
    w2[10, 0] = average
    w2[10, 1] = average
    b2[10] = -1.0

    # Channels 11 and 12: larger-scale midpoint center-surround.
    surround = np.full((3, 3), -1.0 / 8.0, dtype=dtype)
    surround[1, 1] = 1.0
    w2[11, 0] = 0.5 * surround
    w2[11, 1] = -0.5 * surround
    w2[12] = -w2[11]

    # S = M - 0.5*G - 0.25*(horizontal + vertical ridges)
    #       - 0.15*(bright + dark spots).
    # G averages the four opposite-polarity gradient pairs.
    smooth_bright = w2[0].copy()
    for channel in range(2, 10):
        smooth_bright[channel] -= 0.125 * average
    for channel in range(10, 14):
        smooth_bright[channel] -= 0.25 * average
    for channel in range(14, 16):
        smooth_bright[channel] -= 0.15 * average

    # Their difference after pooling is a capped bright-area response.
    w2[13] = smooth_bright
    b2[13] = 0.5 - 0.30
    w2[14] = smooth_bright
    b2[14] = 0.5 - 0.65

    # Triangular dark-area response:
    # ReLU(D-.55) - 2*ReLU(D-.75) + ReLU(D-.95).
    for channel, threshold in zip((15, 16, 17), (0.55, 0.75, 0.95)):
        w2[channel] = w2[1]
        b2[channel] = 0.5 - threshold

    # Channels 18..31 remain exactly zero.
    lung = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.10, 0.19, 0.19, 0.10],
            [0.09, 0.19, 0.05, 0.09],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=dtype,
    )

    # Pass 2 change: head[0] and head[1] remain zero.
    # There is no fixed cardiac-reference brightness comparison.

    # -G: average the four orientation magnitudes.
    for channel in range(2, 6):
        head[channel] = -0.25 * lung

    # Local range and larger-scale signed structure.
    head[10] = -0.5 * lung
    head[11] = 0.5 * lung
    head[12] = -0.5 * lung

    # +2*O: capped smooth-bright response.
    head[13] = 2.0 * lung
    head[14] = -2.0 * lung

    # -2*A: triangular dark-area response.
    head[15] = -2.0 * lung
    head[16] = 4.0 * lung
    head[17] = -2.0 * lung

    # Original reference values for the retained features.
    # Pass 2 removes the former brightness-comparison offset of +0.60.
    bias = (
        -2.0 * 0.10
        + 2.0 * 0.10
        + 0.06
        + 0.5 * 0.08
    )

    return {
        "conv1.weight": w1,
        "conv1.bias": b1,
        "conv2.weight": w2,
        "conv2.bias": b2,
        "fc.weight": head.reshape(1, 512),
        "fc.bias": np.array([bias], dtype=dtype),
    }
