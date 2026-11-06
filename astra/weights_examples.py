# weights_examples.py
import numpy as np


def make_weights():
    """Return six deterministic, hand-designed float32 parameter arrays."""
    w1 = np.zeros((16, 1, 5, 5), dtype=np.float64)
    b1 = np.zeros(16, dtype=np.float64)
    w2 = np.zeros((32, 16, 3, 3), dtype=np.float64)
    b2 = np.zeros(32, dtype=np.float64)

    def unit_lobes(kernel):
        """Zero DC; positive and negative lobes each have unit mass."""
        kernel = np.asarray(kernel, dtype=np.float64)
        kernel = kernel - kernel.mean()
        return kernel / np.maximum(kernel, 0.0).sum()

    # ----- conv1: intensity, derivatives, and ridges -----

    s5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    s3 = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    wide = np.outer(s5, s5)
    narrow = np.outer(s3, s3)

    w1[0, 0] = wide
    w1[1, 0] = -wide
    b1[1] = 1.0

    # Constant support channel. Its weights remain exactly zero.
    b1[2] = 1.0
    w1[3, 0] = narrow

    d5 = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0
    dx5 = np.outer(s5, d5)
    dy5 = dx5.T.copy()
    directions5 = (
        dx5,
        dy5,
        unit_lobes(dx5 + dy5),
        unit_lobes(dx5 - dy5),
    )

    for direction, kernel in enumerate(directions5):
        channel = 4 + 2 * direction
        w1[channel, 0] = kernel
        w1[channel + 1, 0] = -kernel

    ridge_axis = np.array([-1.0, 0.0, 2.0, 0.0, -1.0]) / 2.0
    vertical_ridge = np.outer(s5, ridge_axis)
    horizontal_ridge = vertical_ridge.T.copy()

    w1[12, 0] = vertical_ridge
    w1[13, 0] = -vertical_ridge
    w1[14, 0] = horizontal_ridge
    w1[15, 0] = -horizontal_ridge

    # ----- conv2: combine first-layer pooled features -----

    s = np.array([1.0, 2.0, 1.0]) / 4.0
    blur = np.outer(s, s)

    def add_midpoint(channel, kernel, offset=0.0):
        # U = pooled wide brightness.
        # D = pooled complement = 1 - local minimum brightness.
        # T = constant support channel.
        # M = (U + T - D) / 2 is the local extrema midpoint.
        # Using T instead of a bias preserves zero padding of M.
        w2[channel, 0] += 0.5 * kernel
        w2[channel, 1] -= 0.5 * kernel
        w2[channel, 2] += 0.5 * kernel
        b2[channel] += offset

    add_midpoint(0, blur)
    add_midpoint(1, -blur, offset=1.0)

    # Dark-intensity triangle:
    # relu(0.55 - M) - 2 * relu(0.275 - M).
    add_midpoint(2, -blur, offset=0.55)
    add_midpoint(3, -blur, offset=0.275)

    # Intermediate-brightness trapezoid:
    # relu(M - 0.25) - relu(M - 0.50) - relu(M - 0.75).
    # On [0, 1], this is zero at black and white.
    add_midpoint(4, blur, offset=-0.25)
    add_midpoint(5, blur, offset=-0.50)
    add_midpoint(6, blur, offset=-0.75)

    # Local range U + D - T, averaged spatially.
    w2[7, 0] = blur
    w2[7, 1] = blur
    w2[7, 2] = -blur

    # Broader directional contrast, in first-layer coordinates.
    d3 = np.array([-1.0, 0.0, 1.0])
    dx3 = np.outer(s, d3)
    dy3 = dx3.T.copy()
    directions3 = (
        dx3,
        dy3,
        unit_lobes(dx3 + dy3),
        unit_lobes(dx3 - dy3),
    )

    for direction, kernel in enumerate(directions3):
        channel = 8 + 2 * direction
        add_midpoint(channel, kernel)
        add_midpoint(channel + 1, -kernel)

    # Fine directional energy: combine separately rectified signs.
    for direction in range(4):
        source = 4 + 2 * direction
        w2[16 + direction, source] = blur
        w2[16 + direction, source + 1] = blur

    # Fine contrast from the two pooled blur scales.
    w2[20, 3] = blur
    w2[20, 0] = -blur
    w2[21, 0] = blur
    w2[21, 3] = -blur

    # Preserve four signed ridge responses.
    for index in range(4):
        w2[22 + index, 12 + index] = blur

    # Broad center-surround contrast.
    center = np.zeros((3, 3), dtype=np.float64)
    center[1, 1] = 1.0
    surround = unit_lobes(center - blur)

    add_midpoint(26, surround)
    add_midpoint(27, -surround)

    # Fine positive contrast, penalizing bright linear structures.
    w2[28, 3] = blur
    w2[28, 0] = -blur
    w2[28, 12] = -0.35 * blur
    w2[28, 14] = -0.35 * blur
    b2[28] = -0.015

    # Fine negative contrast, penalizing dark linear structures.
    w2[29, 0] = blur
    w2[29, 3] = -blur
    w2[29, 13] = -0.35 * blur
    w2[29, 15] = -0.35 * blur
    b2[29] = -0.015

    # Broad contrast that survives a fine-edge penalty.
    add_midpoint(30, surround, offset=-0.010)
    add_midpoint(31, -surround, offset=-0.010)

    for channel in (30, 31):
        for source in (4, 5, 6, 7):
            w2[channel, source] -= 0.20 * blur

    # ----- fc: fixed anatomical weighting -----

    lung = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.35, 1.00, 1.00, 0.35],
            [0.45, 1.00, 0.50, 0.45],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    lung /= lung.sum()

    reference = np.zeros((4, 4), dtype=np.float64)
    reference[2, 2] = 1.0
    comparison = lung - reference

    gains = np.zeros(32, dtype=np.float64)
    gains[2:8] = [-3.0, 6.0, 3.5, -3.5, -3.5, -0.50]
    gains[8:16] = 0.30
    gains[16:20] = -0.25
    gains[20:22] = [0.40, -0.20]
    gains[22:26] = -0.075
    gains[26:32] = [1.50, -1.00, 0.75, -0.35, 3.00, -2.00]

    spatial_weights = gains[:, None, None] * lung[None, :, :]
    spatial_weights[0] = 0.20 * comparison
    spatial_weights[1] = -0.20 * comparison

    # Analytic intensity-only crossover at an assumed gray level.
    nominal_gray = 0.42
    nominal_opacity = nominal_gray - 0.25
    nominal_air = 0.55 - nominal_gray
    bias = -(3.5 * nominal_opacity - 3.0 * nominal_air)

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": spatial_weights.reshape(1, 512).astype(np.float32),
        "fc.bias": np.array([bias], dtype=np.float32),
    }