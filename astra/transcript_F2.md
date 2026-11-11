# Build pneumonia classifier weights

> User context
>
> Attachments:
> - This is the same task as below (including the 16 attached example images), with…: `~/.codex/attachments/74500a8b-0362-48d1-bfd5-774685ba7e66/Pasted text.txt`
> Images:
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_01_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_02_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_03_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_04_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_05_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_06_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_07_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_08_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_09_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_10_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_11_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_12_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_13_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_14_NORMAL.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_15_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/example_16_PNEUMONIA.png`
> - `~/OneDrive/Desktop/UAEU/PHD/Advanced_AI/Assignment_2/Astra_project/With Examples/astra_examples/examples_contact_sheet.png`

<details><summary>3 previous messages</summary>

> I’ll read the pasted request first, then use the attached examples to guide the work.
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/attachments/74500a8b-0362-48d1-bfd5-774685ba7e66/Pasted text.txt`
>
> </details>
>
> I’ll submit PRACTICE PASS 1 with a hand-designed baseline for lung brightness, dark air spaces, and texture. The examples vary in exposure and positioning, so those are likely weaknesses. I’ll use fixed formulas without numerical image analysis or testing, then use your diagnostics to guide any revisions.
</details>

**PRACTICE PASS 1 — `weights_diag_v1.py`**

Several pneumonia-labelled examples have brighter, less uniformly dark lung regions and coarser markings. Other examples are subtle. Exposure, positioning, and black padding also vary, so brightness alone is an uncertain signal. This design combines brightness, dark air spaces, and texture using fixed, hand-chosen filters. No numerical measurements or fitting of the examples were used.

**Conv1 filters**

| Channels | What they detect |
|---|---|
| 0–1 | Mean brightness and its complement; together they preserve information about local bright and dark regions after max pooling. |
| 2–3 | Bright and dark small details, comparing the central 3 × 3 region with its surrounding ring. |
| 4–5 | Opposite horizontal intensity gradients. |
| 6–7 | Opposite vertical intensity gradients. |
| 8–9 | Opposite gradients along one diagonal. |
| 10–11 | Opposite gradients along the other diagonal. |
| 12–13 | Bright and dark horizontal lines, including rib-like detail. |
| 14–15 | Bright and dark vertical lines. |

**Conv2 channels**

Here, \(M\) is the reconstructed local brightness; \(T\) combines the two small-detail filters; \(H,V\) combine the horizontal and vertical line filters; and \(E\) averages the four gradient magnitudes. Except for channels 24–29, combinations use a 3 × 3 mean. The expressions below describe values **before ReLU and the second max pool**.

| Channels | Combination |
|---|---|
| 0–1 | \(M\), \(1-M\). |
| 2–5 | Brightness above thresholds 0.25, 0.45, 0.65, and 0.80. |
| 6–7 | Darkness: \(0.65-M\), \(0.45-M\). |
| 8 | Local intensity range retained by conv1 pooling. |
| 9 | Small-detail magnitude \(T\). |
| 10–13 | Horizontal, vertical, and two diagonal gradient magnitudes. |
| 14–15 | Horizontal and vertical line magnitudes \(H,V\). |
| 16–17 | Smooth brightness: \(M-0.5E\), above thresholds 0.25 and 0.45. |
| 18–19 | Smooth darkness: \(0.65-M-0.5E\), \(0.45-M-0.5E\). |
| 20 | Small detail with horizontal-line suppression: \(T-0.5H\). |
| 21–22 | Vertical-line excess and horizontal-line excess. |
| 23 | Diagonal-gradient excess over vertical gradients. |
| 24–25 | Coarser bright and dark centre-surround features. |
| 26–29 | Coarser directional gradients, with both polarities. |
| 30–31 | Small detail combined with brightness or darkness. |

The final layer uses this spatial mask, divided by **10.25**:

| Grid row | Column 0 | Column 1 | Column 2 | Column 3 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 2 | 1.5 | 1 |
| 2 | 1 | 2 | 0.75 | 1 |
| 3 | 0 | 0 | 0 | 0 |

This emphasizes the middle chest. The lower weight in row 2, column 2 reduces influence from the cardiac silhouette. Peripheral columns receive smaller weights because they also contain chest wall and padding.

Channel multipliers are **0: +1.5, 1: −1.5, 2: +2, 4: −2, 7: −3, 16: +2, 17: −2, 20: +6, 22: −3**. These reward brightness and smooth opacity, penalize dark regions, and weakly favor non-horizontal texture over rib-like texture. Paired threshold channels cap the brightness contributions. All other channels initially have zero final-layer weights and serve as diagnostic probes.

The bias is **−0.50**, a manually chosen offset that puts an illustrative moderate-brightness, moderate-texture response near the decision boundary. It is not calibrated from data.

My **unvalidated AUROC prediction is 0.60**, with approximately **0.55** on images from another hospital.

```python
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
```

> User context
>
> Attachments:
> - OK: astra/practice_F2/weights_diag_v1.py -> 5,569 parameters, all shapes correc…: `~/.codex/attachments/2033d76c-911b-4c46-825f-cb51f003c862/Pasted text.txt`

<details><summary>2 previous messages</summary>

> I’ll read the practice results and use the diagnostics to decide whether a hand-designed revision is justified.
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/attachments/2033d76c-911b-4c46-825f-cb51f003c862/Pasted text.txt`
>
> </details>
</details>

**PRACTICE PASS 2 — `weights_diag_v2.py`**

Pass 1 achieved **AUROC 0.5890**. The clearest diagnostic finding is that local contrast is consistently **lower on pneumonia-labelled images**. Conv2 channel 8, which measures local intensity range, has separation **0.061**. Reversing its ranking corresponds to approximately **0.939 on that same batch**; this is a deduction from your diagnostic, not a new evaluation.

I will test a simple revision: **keep every convolutional parameter unchanged and use only channel 8, with weight −1 on each of its 16 grid cells**. Uniform spatial weighting matches the whole-image statistic reported in the diagnostics. This also removes channel 20’s positive contribution, which was larger for normal images.

The examples motivated brightness and texture detectors, but the feedback favors reduced contrast much more strongly than my original combination. This association could include exposure, padding, or acquisition differences; its transfer to another hospital remains uncertain.

The unchanged filters are:

| Conv1 channels | Detection |
|---|---|
| 0–1 | Smoothed brightness and darkness |
| 2–3 | Bright and dark centre-surround detail |
| 4–7 | Horizontal and vertical gradients, both polarities |
| 8–11 | Two diagonal gradients, both polarities |
| 12–15 | Horizontal and vertical lines, both polarities |

| Conv2 channels | Combination |
|---|---|
| 0–1 | Reconstructed brightness and darkness |
| 2–5 | Brightness thresholds: 0.25, 0.45, 0.65, 0.80 |
| 6–7 | Darkness thresholds |
| **8** | **Local range: averaged maximum brightness minus minimum brightness from the first pooling stage** |
| 9–15 | Small-detail, directional-gradient, and line magnitudes |
| 16–19 | Brightness/darkness with an edge penalty |
| 20–23 | Relative texture and orientation features |
| 24–29 | Coarser centre-surround and directional contrasts |
| 30–31 | Detail combined with brightness or darkness |

All other channels retain zero final-layer weights and remain available for diagnostics. The new logit is exactly:

\[
z=1-\sum_{r=0}^{3}\sum_{c=0}^{3}h_{8,r,c}.
\]

The **+1 bias and −1 cell weights are fixed unit constants**, not formulas of the reported statistics. The bias gives a transparent initial threshold; the probabilities remain uncalibrated.

My **next-batch AUROC prediction is 0.90**. This is a candidate revision; whether to retain it depends on the next practice result, remembering that the batch changes.

```python
# weights_diag_v2.py
# PRACTICE PASS 2
#
# Pass 1: Hand-designed brightness and texture combination.
#         Reported AUROC: 0.5890.
# Pass 2: Keep all convolutional parameters unchanged.
#         Replace the final layer with uniform negative weighting
#         of conv2 channel 8, the local intensity-range detector.
#         Use fixed unit coefficients and a fixed unit intercept.
#
# No training, fitting, random numbers, or data-dependent constants.

import numpy as np

def make_weights():
    """Return the six float32 parameter arrays for WrittenCNN."""
    dtype = np.float32

    conv1_weight = np.zeros((16, 1, 5, 5), dtype=dtype)
    conv1_bias = np.zeros(16, dtype=dtype)
    conv2_weight = np.zeros((32, 16, 3, 3), dtype=dtype)
    conv2_bias = np.zeros(32, dtype=dtype)
    fc_weight = np.zeros((1, 512), dtype=dtype)
    fc_bias = np.array([1.0], dtype=dtype)

    # ----- Conv1: unchanged from pass 1 -----

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

    # ----- Conv2: unchanged from pass 1 -----

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
    # Uniform negative contribution from all 16 channel-8 cells.
    # All other channels remain diagnostic probes.
    # Flattening order: channel * 16 + row * 4 + column.
    fc_weight[0, 8 * 16 : 9 * 16] = -1.0

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }
```

> User context
>
> Attachments:
> - OK: astra/practice_F2/weights_diag_v1.py -> 5,569 parameters, all shapes correc…: `~/.codex/attachments/83541666-0e06-45cf-b11f-bab26825d435/Pasted text.txt`

<details><summary>2 previous messages</summary>

> I’ll read the latest results and check whether the contrast-based revision improved performance.
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/attachments/83541666-0e06-45cf-b11f-bab26825d435/Pasted text.txt`
>
> </details>
</details>

This attachment repeats **PRACTICE PASS 1** for `weights_diag_v1.py`: AUROC **0.5890**, with `fc.bias = −0.5`. It does not contain results for v2.

Please run the previously supplied **`weights_diag_v2.py` as PRACTICE PASS 2** and return its results. Its expected parameter checks are:

- `fc.bias = +1.0`
- `fc.weight`: exactly **16 entries equal to −1**, at indices **128–143**; all others zero.
- All convolutional parameters unchanged from v1.

I’m keeping v2 unchanged pending that evaluation; no additional revision is being submitted.

> User context
>
> Attachments:
> - OK: astra/practice_F2/weights_diag_v2.py -> 5,569 parameters, all shapes correc…: `~/.codex/attachments/93f64b03-ad89-4f44-8387-647cbcce30a6/Pasted text.txt`

<details><summary>3 previous messages</summary>

> I’ll read pass 2’s results and check whether the contrast signal held up on the next batch.
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/attachments/93f64b03-ad89-4f44-8387-647cbcce30a6/Pasted text.txt`
>
> </details>
>
> Pass 2 reached AUROC 0.9331, supporting the contrast signal on a second batch. Its 96.92% sensitivity and 37.70% specificity show that the current threshold favors pneumonia predictions. I’ll keep the ranking function and test a small, fixed bias reduction to improve specificity.
</details>

**PRACTICE PASS 3 — `weights_diag_v3.py`**

Pass 2 achieved **AUROC 0.9331**, supporting the contrast detector on a second batch. However, **sensitivity was 0.9692 and specificity 0.3770**: the decision threshold favors pneumonia predictions.

I will retain the entire feature design and change **only `fc.bias` from +1.00 to +0.90**. This is a small, hand-chosen correction of 0.10 logit units, not a value calculated from class means or per-image scores.

On identical images, this change preserves every ranking and therefore preserves AUROC. It can increase specificity at the expense of sensitivity. The next practice batch will show whether that tradeoff is useful.

The examples originally motivated brightness, dark-space, and texture detectors. The practice diagnostics support reduced local contrast most strongly. The unchanged filters are:

| Conv1 channels | Detection |
|---|---|
| 0–1 | Smoothed brightness and darkness |
| 2–3 | Bright and dark centre-surround detail |
| 4–7 | Horizontal and vertical gradients, both polarities |
| 8–11 | Two diagonal gradients, both polarities |
| 12–15 | Horizontal and vertical lines, both polarities |

| Conv2 channels | Combination |
|---|---|
| 0–1 | Reconstructed brightness and darkness |
| 2–5 | Brightness above four fixed thresholds |
| 6–7 | Darkness below two fixed thresholds |
| **8** | **Local intensity range retained by the first pooling stage** |
| 9–15 | Detail, directional-gradient, and line magnitudes |
| 16–19 | Brightness/darkness with an edge penalty |
| 20–23 | Relative texture and orientation features |
| 24–29 | Coarser centre-surround and directional contrasts |
| 30–31 | Detail combined with brightness or darkness |

The final layer still assigns **−1 to all 16 grid cells of channel 8**, with every other weight zero. Uniform weighting matches the whole-image contrast statistic supported by the diagnostics. The revised logit is:

\[
z=0.90-\sum_{r=0}^{3}\sum_{c=0}^{3}h_{8,r,c}.
\]

My **next-batch AUROC prediction is approximately 0.93**. The probabilities remain uncalibrated, and cross-hospital performance is still unknown.

```python
# weights_diag_v3.py
# PRACTICE PASS 3
#
# Pass 1: Hand-designed brightness and texture combination.
#         Reported AUROC: 0.5890.
# Pass 2: Uniform negative weighting of conv2 channel 8.
#         Reported AUROC: 0.9331.
# Pass 3: Change only fc.bias from +1.00 to +0.90.
#         A fixed, modest correction toward greater specificity.
#
# No training, fitting, random numbers, or data-dependent constants.

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

    # ----- Conv1: unchanged from passes 1 and 2 -----

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

    # ----- Conv2: unchanged from passes 1 and 2 -----

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

    # ----- Final-layer weights: unchanged from pass 2 -----
    # Uniform negative contribution from all 16 channel-8 cells.
    # All other channels remain diagnostic probes.
    # Flattening order: channel * 16 + row * 4 + column.
    fc_weight[0, 8 * 16 : 9 * 16] = -1.0

    return {
        "conv1.weight": conv1_weight,
        "conv1.bias": conv1_bias,
        "conv2.weight": conv2_weight,
        "conv2.bias": conv2_bias,
        "fc.weight": fc_weight,
        "fc.bias": fc_bias,
    }
```

> User context
>
> Attachments:
> - OK: astra/practice_F2/weights_diag_v3.py -> 5,569 parameters, all shapes correc…: `~/.codex/attachments/887e4143-4630-47e7-acfa-bd307351efdf/Pasted text.txt`

<details><summary>2 previous messages</summary>

> I’ll read pass 3’s results and check how the bias change affected sensitivity and specificity.
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/attachments/887e4143-4630-47e7-acfa-bd307351efdf/Pasted text.txt`
>
> </details>
</details>

**PRACTICE PASS 4 — `weights_diag_v4.py`**

Pass 3 achieved **AUROC 0.9177, sensitivity 0.9451, and specificity 0.5811**. Its lower AUROC than pass 2 reflects the different batch: changing only an additive bias preserves rankings.

I’ll retain the **+0.90 bias** and test one architectural combination suggested by the diagnostics: **fine local contrast plus broader horizontal contrast**. Channels 26 and 27 consistently show the same class direction as channel 8, and both separate slightly better than channel 8 on this batch.

The new final-layer weights, applied uniformly across all 16 cells, are:

| Channel | Weight per cell | Purpose |
|---|---:|---|
| 8 | −0.50 | Retain fine local intensity range |
| 26 | −0.25 | Add broader horizontal contrast of one polarity |
| 27 | −0.25 | Add the opposite polarity |
| All others | 0 | Retain as diagnostic probes |

The **half-and-half split between scales**, with equal weights for the two coarse polarities, is a fixed design choice. No coefficient is calculated from separations, activation means, or per-image scores. Whether these features provide complementary information remains to be tested.

All convolutional parameters remain unchanged. The examples motivated the following brightness and texture measurements:

| Conv1 channels | Detection |
|---|---|
| 0–1 | Smoothed brightness and darkness |
| 2–3 | Bright and dark centre-surround detail |
| 4–7 | Horizontal and vertical gradients, both polarities |
| 8–11 | Two diagonal gradients, both polarities |
| 12–15 | Horizontal and vertical lines, both polarities |

| Conv2 channels | Combination |
|---|---|
| 0–1 | Reconstructed brightness and darkness |
| 2–5 | Brightness thresholds: 0.25, 0.45, 0.65, 0.80 |
| 6–7 | Darkness below fixed thresholds |
| 8 | Fine local intensity range |
| 9–15 | Detail, directional-gradient, and line magnitudes |
| 16–19 | Brightness/darkness with an edge penalty |
| 20–23 | Relative texture and orientation features |
| 24–25 | Coarser centre-surround contrast |
| 26–29 | Coarser horizontal and vertical gradients, both polarities |
| 30–31 | Detail combined with brightness or darkness |

Uniform spatial weighting retains the whole-image measurements supported by the diagnostics. The score becomes:

\[
z=0.90-\sum_{r,c}
\left(0.50h_{8,r,c}+0.25h_{26,r,c}+0.25h_{27,r,c}\right).
\]

My **practice AUROC prediction is approximately 0.93**. This combination is a candidate; v3 remains the reference until its results arrive.

```python
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
```

> User context
>
> Attachments:
> - OK: astra/practice_F2/weights_diag_v4.py -> 5,569 parameters, all shapes correc…: `~/.codex/attachments/430528eb-d94e-4f1e-a353-cea0b44eeb5d/Pasted text.txt`

<details><summary>2 previous messages</summary>

> I’ll read pass 4’s results and use the channel diagnostics to compare the combined contrast score with the previous detector on the same batch.
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/attachments/430528eb-d94e-4f1e-a353-cea0b44eeb5d/Pasted text.txt`
>
> </details>
</details>

**FINAL — `weights_diag_final.py`**

I’m submitting **exactly the weights from pass 4**, with no further changes.

Pass 4 achieved **AUROC 0.9377, accuracy 0.8867, sensitivity 0.9005, and specificity 0.8462**. Its AUROC essentially matches the approximately **0.937** implied by reversing channel 8 alone on that same batch. I’m retaining the combination for its observed sensitivity/specificity balance; the ranking improvement is negligible.

All four practice passes are recorded below. They used different, cycling batches, so their aggregate metrics are not paired comparisons.

| Pass | Change and reason | AUROC |
|---|---|---:|
| 1 | Built brightness, darkness, and texture filters with middle-chest emphasis, based on the examples’ apparent opacity and texture differences. | 0.5890 |
| 2 | Replaced the final layer with uniform negative weights on local-range channel 8. Diagnostics consistently showed lower contrast on pneumonia-labelled images. Set a fixed +1 bias. | 0.9331 |
| 3 | Reduced the bias to +0.90 using a fixed 0.10-logit correction to reduce false positives. Rankings were unchanged. | 0.9177 |
| 4 | Combined fine local range with broader horizontal contrast, splitting weight equally between scales and equally between the two coarse polarities. Retained bias +0.90. | 0.9377 |

The examples showed differences in lung brightness and texture, alongside substantial exposure and positioning variation. Those observations motivated the filter bank. Practice feedback favored **reduced contrast** over the original brightness-based combination.

The conv1 filters detect:

| Channels | Detection |
|---|---|
| 0–1 | Five-by-five mean brightness and its complement |
| 2–3 | Bright and dark central detail relative to a surrounding ring |
| 4–5 | Horizontal gradients, opposite polarities |
| 6–7 | Vertical gradients, opposite polarities |
| 8–9 | First diagonal gradients, opposite polarities |
| 10–11 | Second diagonal gradients, opposite polarities |
| 12–13 | Bright and dark horizontal lines |
| 14–15 | Bright and dark vertical lines |

The conv2 channels combine them as follows. Here, \(M\) denotes reconstructed local brightness, \(T\) small-detail magnitude, \(E\) average gradient magnitude, and \(H,V\) horizontal and vertical line magnitudes. Thresholds apply before ReLU.

| Channels | Combination |
|---|---|
| 0–1 | \(M\) and \(1-M\) |
| 2–5 | \(M\) above thresholds 0.25, 0.45, 0.65, 0.80 |
| 6–7 | \(0.65-M\), \(0.45-M\) |
| 8 | Local intensity range retained by the first pooling stage |
| 9 | Small-detail magnitude \(T\) |
| 10–13 | Four directional gradient magnitudes |
| 14–15 | Line magnitudes \(H,V\) |
| 16–17 | \(M-0.5E\) above thresholds 0.25 and 0.45 |
| 18–19 | \(0.65-M-0.5E\), \(0.45-M-0.5E\) |
| 20–23 | Detail minus horizontal lines; vertical-line excess; horizontal-line excess; diagonal-gradient excess |
| 24–25 | Coarser bright and dark centre-surround contrast |
| 26–27 | Coarser horizontal gradients, opposite polarities |
| 28–29 | Coarser vertical gradients, opposite polarities |
| 30–31 | Detail combined with brightness or darkness |

Only channels **8, 26, and 27** affect the output. Each of their 16 grid cells receives weight **−0.50, −0.25, and −0.25**, respectively. Uniform weighting matches the whole-image measurements supported by the diagnostics. All other final-layer weights are zero.

The bias remains **+0.90**, giving:

\[
z=0.90-\sum_{r,c}\left(0.50h_{8,r,c}+0.25h_{26,r,c}+0.25h_{27,r,c}\right).
\]

My AUROC forecast is **approximately 0.92 on similar data**. For another hospital, my much less certain guess is **0.65**: this global contrast score may also reflect acquisition and padding differences. Neither forecast is a measured scoring-set result.

```python
# weights_diag_final.py
# FINAL
#
# Exactly the weights evaluated in PRACTICE PASS 4.
#
# Practice history:
# 1. Brightness/texture combination: AUROC 0.5890.
# 2. Negative global local-range score: AUROC 0.9331.
# 3. Bias reduced from +1.00 to +0.90: AUROC 0.9177.
# 4. Fixed fine/coarse contrast combination: AUROC 0.9377.
#
# Passes used different, cycling practice batches.
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

    # ----- Conv1 -----

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

    # ----- Conv2 -----

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

    # ----- Final layer -----
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
```
