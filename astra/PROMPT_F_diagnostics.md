# Prompt F — 16 example images + up to 20 practice passes WITH channel diagnostics

Condition F = condition D plus one change: every practice pass also reports how each channel of the
network behaves on the practice images (16 conv1 + 32 conv2 rows). Question: does richer feedback, not
just a score, let an LLM write better weights?

Start a NEW Astra chat for each run (it must not see conversations A–E or another F run). Attach the
same 17 files as in conditions C and D (the 16 `example_…png` files and `examples_contact_sheet.png`),
then paste everything below the line. Each time Astra sends a "PRACTICE PASS n" file, run notebook cell
F2 and paste back its full output. Never paste anything from the test set. When Astra says FINAL, send
the file and the conversation for checking before cell F3 scores it.

---

This is the same task as below (including the 16 attached example images), with one difference. You may run up to **20 practice forward passes** before you submit your final weights. A practice pass works like this: you send me a complete weights file; I run your network once, with no training, on 256 practice images of children who are NOT in the scoring set (each pass uses the next 256 of 782 practice images, cycling). I send back:

1. AUROC, accuracy, sensitivity, specificity and the mean P(pneumonia) for each class;
2. a per-image table of practice-image id, true label and your network's P(pneumonia);
3. **CHANNEL DIAGNOSTICS**: for each of the 16 conv1 channels and each of the 32 conv2 channels, its mean activation on normal and on pneumonia images, its *separation* (the AUROC of that channel's mean activation on its own: 0.5 means it does not tell the classes apart, above 0.5 means it is higher on pneumonia, below 0.5 higher on normal), and the share of images on which it is *silent* (zero everywhere). For each conv2 channel you also get its mean contribution to the logit (its 16 fc weights times its 16 grid-cell values) on normal and on pneumonia images.

The diagnostics are there so you can debug your own design, the way an engineer debugs a hand-built detector: find filters that are silent or saturated, filters that respond to the wrong thing, channels whose contribution to the logit has the wrong sign, and channels that carry no signal. You will never see the practice images themselves (only the 16 attached examples).

Rules for the practice passes:
- Every change to the weights must be one you design and can explain (for example "conv2 channel 7 is silent on 100 % of images because its bias is too negative, so I raise it to …", or "conv1 filter 3 separates the classes in the wrong direction, so I flip the sign of its fc weights"). Do not run any optimisation, gradient estimation, finite differences, random search or automated loop over the practice outputs.
- Do not set weights by fitting them to the reported numbers: no weight may be computed as a formula of the per-image scores or of the diagnostic statistics (for example, setting each fc weight proportional to its channel's separation is not allowed). Use the diagnostics to decide *what* to fix, then design the fix.
- Keep a change only if it helps, as you judge from the practice results.
- Say "PRACTICE PASS n" with each file you want run, and "FINAL" with the file to be scored. The scoring set is used exactly once, on the FINAL file.
- With your FINAL file, list every pass you used and what you changed and why.

---

You will write, by hand, every weight of a small convolutional neural network that detects pneumonia in frontal chest X-rays. No training of any kind will happen: no loss function, no gradients, no optimisation, no tools, and no feedback other than the practice passes (scores and channel diagnostics) described above. Your weights will be loaded exactly as you write them and scored once on X-rays you will never see, including X-rays from a different hospital.

## What you can look at

Attached are 16 labelled example X-rays: 8 NORMAL and 8 PNEUMONIA (the label is in each file name; `examples_contact_sheet.png` shows all 16 together). Each one is exactly what the network receives as input. Use them to see what the inputs look like (brightness, contrast, where the lungs sit in the frame, what pneumonia looks like at this resolution) and design your weights from that, together with any medical, radiological and image-processing knowledge you have. Do not try to fit weights to these 16 images numerically; they are examples to look at, not training data.

## How every image is prepared before it reaches the network

1. The original X-ray (any size or aspect ratio) is converted to single-channel grayscale, 8-bit.
2. It is padded to a square with black (value 0) bars, centred, on the shorter side.
3. The square is resized to 224 × 224 pixels.
4. The network receives it as values in [0, 1]: 0 = black, 1 = white.

No augmentation, flipping or cropping is applied at scoring time.

## The network (PyTorch, exactly as it will run)

```python
class WrittenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)    # zero padding
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)   # zero padding
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(512, 1)

    def forward(self, g):                          # g: [B, 1, 224, 224], values in [0, 1]
        h = F.max_pool2d(F.relu(self.conv1(g)), 2) # [B, 16, 56, 56]
        h = F.max_pool2d(F.relu(self.conv2(h)), 2) # [B, 32, 28, 28]
        h = self.pool(h)                            # [B, 32, 4, 4]: the mean over a 4 x 4 grid of 7 x 7 cells
        h = torch.flatten(h, 1)                     # [B, 512]; index = channel * 16 + row * 4 + col
        return self.fc(h)                           # logit; P(pneumonia) = sigmoid(logit)
```

Parameter shapes (5,569 numbers in total):

| name | shape |
|---|---|
| conv1.weight | [16, 1, 5, 5] |
| conv1.bias | [16] |
| conv2.weight | [32, 16, 3, 3] |
| conv2.bias | [32] |
| fc.weight | [1, 512] |
| fc.bias | [1] |

Conventions: conv weights are [out_channel, in_channel, kernel_row, kernel_col]; PyTorch convolution is cross-correlation (no kernel flip). Row 0 of the 4 × 4 grid is the top of the image; column 0 is the image's left side, which by radiographic convention is the patient's right.

## How your weights will be scored

Primary metric: AUROC. Also reported: accuracy at P = 0.5, sensitivity (pneumonia recall) and specificity (normal recall). Your network will be compared with the same architecture trained by gradient descent and with larger trained networks, on held-out X-rays and on X-rays from a different hospital.

## What to send back

One Python file, and nothing else in a code block:

```python
# weights_diag_vN.py  (or weights_diag_final.py)
import numpy as np
import math

def make_weights():
    """Return {name: array} with exactly the shapes above."""
    ...
    return {
        "conv1.weight": ..., "conv1.bias": ...,
        "conv2.weight": ..., "conv2.bias": ...,
        "fc.weight": ..., "fc.bias": ...,
    }
```

Rules for the file: it may import only `numpy` and `math`; it must not read or write files, use the network, or use random numbers (every value must follow from a formula or constant you chose); it must return exactly the six arrays above.

Before the code, explain in plain language: what you saw in the example images that shaped your design, what each conv1 filter detects, what each conv2 channel combines, which 4 × 4 grid cells and channels the final layer weighs and why, and how you set the bias. Then give your own prediction of the AUROC you expect.
