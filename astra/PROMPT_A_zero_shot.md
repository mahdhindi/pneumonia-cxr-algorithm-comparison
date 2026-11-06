# Prompt A — write the weights in one reply (zero-shot)

Copy everything below the line into a NEW Astra conversation, unchanged.
Attach nothing. Do not answer follow-up questions with data or results; if Astra
asks something, answer only from this text ("everything you need is in the task").

---

You will write, by hand, every weight of a small convolutional neural network. No training of any kind will happen: no data, no loss function, no gradients, no optimisation, no tools, and no feedback before your weights are scored. Your weights will be loaded exactly as you write them and scored once on images you will never see. Use your reasoning and any medical, radiological and image-processing knowledge you have to decide what each part of the network should compute, and then write weights that compute it.

## The task

Each input is one frontal chest X-ray of a child aged one to five, from a paediatric hospital. Each image is either NORMAL (label 0) or PNEUMONIA (label 1; bacterial or viral pneumonia, as diagnosed by physicians). The network must output a score whose sigmoid is P(pneumonia). About three quarters of the images available for development are pneumonia.

## How every image is prepared before it reaches the network (this is the actual code path)

1. The original image (a JPEG of varying size and aspect ratio, a few hundred to about three thousand pixels per side) is converted to single-channel grayscale, 8-bit.
2. It is padded to a square with black (value 0) bars, centred, on the shorter side.
3. The square is resized to 224 × 224 pixels.
4. The network receives it as values in [0, 1]: 0 = black, 1 = white. As in all radiographs, air (including the normal lungs) is dark and dense tissue, bone, fluid and consolidation are bright.

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

On a held-out set of children who appear nowhere else, collected separately from the development images. Primary metric: AUROC. Also reported: accuracy at P = 0.5, sensitivity (pneumonia recall) and specificity (normal recall). Your network will be compared with the same architecture trained by gradient descent and with larger trained networks.

## What to send back

One Python file, and nothing else in a code block:

```python
# weights_zeroshot.py
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

Rules for the file: it may import only `numpy` and `math`; it must not read or write files, use the network, or use random numbers (every value must follow from a formula or constant you chose); it must return exactly the six arrays above. Writing weights with loops and formulas (for example a Gaussian or an oriented edge kernel) is encouraged.

Before the code, explain in plain language: what each conv1 filter detects, what each conv2 channel combines, which 4 × 4 grid cells and channels the final layer weighs and why, and how you set the bias. Then give your own prediction of the AUROC you expect.
