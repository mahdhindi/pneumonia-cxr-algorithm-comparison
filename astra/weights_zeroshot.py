# weights_zeroshot.py
import numpy as np
import math


def make_weights():
    """Return the six deterministic parameter arrays as float32."""

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=np.float64)
    conv1_bias = np.zeros(16, dtype=np.float64)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=np.float64)
    conv2_bias = np.zeros(32, dtype=np.float64)
    fc_grid = np.zeros((32, 4, 4), dtype=np.float64)

    coordinates = np.arange(-2, 3, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")

    def gaussian(sigma):
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        return kernel / kernel.sum()

    def contrast_kernel(kernel):
        # Zero DC response; positive and negative lobes each have mass one.
        kernel = kernel - kernel.mean()
        return kernel / np.maximum(kernel, 0.0).sum()

    # conv1 0-3: complementary intensity measurements at two scales.
    gaussian_blur = gaussian(1.0)
    box_blur = np.full((5, 5), 1.0 / 25.0, dtype=np.float64)

    conv1_weight[0, 0] = gaussian_blur
    conv1_weight[1, 0] = -gaussian_blur
    conv1_bias[1] = 1.0

    conv1_weight[2, 0] = box_blur
    conv1_weight[3, 0] = -box_blur
    conv1_bias[3] = 1.0

    # conv1 4-11: directional gradients, both polarities.
    diagonal = math.sqrt(0.5)
    directions = (
        (1.0, 0.0),
        (diagonal, diagonal),
        (0.0, 1.0),
        (-diagonal, diagonal),
    )

    for index, (dx, dy) in enumerate(directions):
        kernel = contrast_kernel((dx * xx + dy * yy) * gaussian_blur)
        channel = 4 + 2 * index
        conv1_weight[channel, 0] = kernel
        conv1_weight[channel + 1, 0] = -kernel

    # conv1 12-13: fine bright/dark center-surround responses.
    fine_kernel = contrast_kernel(gaussian(0.65) - gaussian(1.4))
    conv1_weight[12, 0] = fine_kernel
    conv1_weight[13, 0] = -fine_kernel

    # conv1 14: broader bright center versus its surrounding ring.
    broad_kernel = np.full((5, 5), -1.0 / 16.0, dtype=np.float64)
    broad_kernel[1:4, 1:4] = 1.0 / 9.0
    conv1_weight[14, 0] = contrast_kernel(broad_kernel)

    # conv1 15: constant feature, including at image boundaries.
    conv1_bias[15] = 1.0

    # All following kernels operate on the first pooled feature maps.
    smooth = np.array(
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 2.0],
         [1.0, 2.0, 1.0]],
        dtype=np.float64,
    ) / 16.0

    # If a=max(I) and b=max(1-I), (a-b+1)/2 is the
    # midpoint of the local maximum and minimum intensity.
    # Average this estimate across the two blur scales.
    brightness = np.zeros((16, 3, 3), dtype=np.float64)
    brightness[0] = 0.25 * smooth
    brightness[1] = -0.25 * smooth
    brightness[2] = 0.25 * smooth
    brightness[3] = -0.25 * smooth
    brightness[15] = 0.50 * smooth

    edges = np.zeros_like(brightness)
    for channel in range(4, 12):
        edges[channel] = 0.25 * smooth

    bright_broad = np.zeros_like(brightness)
    bright_broad[14] = smooth

    dark_fine = np.zeros_like(brightness)
    dark_fine[13] = smooth

    fine_texture = np.zeros_like(brightness)
    fine_texture[12] = 0.50 * smooth
    fine_texture[13] = 0.50 * smooth

    # Symmetric knots allow the reversed ordinates to represent f(1-x).
    knots = np.array(
        [0.00, 0.08, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92, 1.00],
        dtype=np.float64,
    )
    ordinates = np.array(
        [0.00, -0.25, -1.00, -0.80, 0.00, 0.65, 0.75, 0.30, 0.15],
        dtype=np.float64,
    )

    def hinge_coefficients(values):
        # f(x) = f(0) + sum_j coefficient[j] * ReLU(x-knots[j])
        # for x in [0, 1].
        slopes = np.diff(values) / np.diff(knots)
        coefficients = np.empty(8, dtype=np.float64)
        coefficients[0] = slopes[0]
        coefficients[1:] = np.diff(slopes)
        return coefficients

    forward_coefficients = hinge_coefficients(ordinates)
    reverse_coefficients = hinge_coefficients(ordinates[::-1])

    lung_weights = np.array(
        [[0.00, 0.10, 0.10, 0.00],
         [0.65, 1.00, 1.00, 0.65],
         [0.65, 0.90, 0.50, 0.65],
         [0.00, 0.05, 0.00, 0.00]],
        dtype=np.float64,
    )
    lung_weights /= lung_weights.sum()

    gain = 6.0
    upper_fraction = 0.30
    lower_fraction = 0.45
    edge_discounted_fraction = 0.25
    edge_discounted_brightness = brightness - 0.55 * edges

    for index, threshold in enumerate(knots[:-1]):
        # conv2 0-7: hinges of brightness.
        conv2_weight[index] = brightness
        conv2_bias[index] = -threshold
        fc_grid[index] += (
            gain * upper_fraction
            * forward_coefficients[index] * lung_weights
        )

        # conv2 8-15: hinges of complementary brightness.
        # Max pooling these retains minima of brightness.
        conv2_weight[8 + index] = -brightness
        conv2_bias[8 + index] = 1.0 - threshold
        fc_grid[8 + index] += (
            gain * lower_fraction
            * reverse_coefficients[index] * lung_weights
        )

        # conv2 16-23: brightness with sharp-edge evidence discounted.
        conv2_weight[16 + index] = edge_discounted_brightness
        conv2_bias[16 + index] = -threshold
        fc_grid[16 + index] += (
            gain * edge_discounted_fraction
            * forward_coefficients[index] * lung_weights
        )

    # Each pair implements amplitude * clip((response-low)/(high-low), 0, 1).
    # The identity remains valid after max pooling because both hinges
    # are monotone functions of the same underlying response.
    bounded_features = (
        (24, bright_broad - 0.60 * edges, 0.015, 0.065, 0.65),
        (26, dark_fine - 0.50 * edges,    0.015, 0.065, -0.20),
        (28, fine_texture - 0.35 * edges, 0.010, 0.060, 0.25),
        (30, edges,                       0.080, 0.200, -0.20),
    )

    for channel, kernel, low, high, amplitude in bounded_features:
        conv2_weight[channel] = kernel
        conv2_weight[channel + 1] = kernel
        conv2_bias[channel] = -low
        conv2_bias[channel + 1] = -high

        coefficient = amplitude / (high - low)
        fc_grid[channel] += coefficient * lung_weights
        fc_grid[channel + 1] -= coefficient * lung_weights

    # Additional relative-brightness evidence:
    # 2 * (lung-region brightness - central-chest brightness).
    reference_weights = np.array(
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.2, 0.2, 0.0],
         [0.0, 0.2, 0.4, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    reference_weights /= reference_weights.sum()
    spatial_difference = lung_weights - reference_weights

    # Channel 0 measures max brightness; channel 8 measures 1-min brightness.
    # The constant cancels because spatial_difference sums to zero.
    fc_grid[0] += spatial_difference
    fc_grid[8] -= spatial_difference

    # f(0)=0 for the forward spline; the reversed spline starts at f(1).
    output_bias = (
        math.log(3.0)
        + gain * lower_fraction * float(ordinates[-1])
    )

    return {
        "conv1.weight": conv1_weight.astype(np.float32),
        "conv1.bias": conv1_bias.astype(np.float32),
        "conv2.weight": conv2_weight.astype(np.float32),
        "conv2.bias": conv2_bias.astype(np.float32),
        "fc.weight": fc_grid.reshape(1, 512).astype(np.float32),
        "fc.bias": np.array([output_bias], dtype=np.float32),
    }