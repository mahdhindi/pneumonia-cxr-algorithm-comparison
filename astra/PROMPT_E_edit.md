# Prompt E — edit the final layer of a trained ResNet-50 (no gradients)

Start a NEW Astra chat. Attach `resnet50_fc.txt` (downloaded by notebook cell E1). Paste everything
below the line, and then, at the end of the same message, paste the full output of cell E1's
"PASS 0" (the unedited model on all practice images). For each "PRACTICE PASS n" file, run cell E2 and
paste back its full output. Never paste anything from the scoring half.

Practice X-rays = one half of the official test folder (split by patient, seed 0, 308 images); scoring =
the other half (310 images). Our validation set is NOT used: the model is already near-perfect on it
(AUROC 0.9995), so the problem the edit should fix does not show there.

---

Below is a trained neural network that detects pneumonia in frontal chest X-rays. It was trained normally (gradient descent) and it works, but not as well as it could on new X-rays. Your job is to improve it by editing its final layer by hand. No training of any kind will happen: no loss function, no gradients, no optimisation algorithm, and no fitting of numbers to data.

## The network

A ResNet-50 (pretrained on everyday photographs, then fine-tuned on chest X-rays). For each X-ray, the network's body produces 2,048 features (each a number ≥ 0, after ReLU and global average pooling). Its final layer turns them into a prediction:

    logit = sum over i of weight[i] * feature[i]  +  bias
    P(pneumonia) = sigmoid(logit)

You may change only this final layer: the 2,048 weights and the bias. The attached file `resnet50_fc.txt` lists every current weight (index, value) and the bias. The 2,048 features have no names; what they respond to is unknown.

## What you get

- Below this prompt: **PASS 0**, the unedited network's results on all practice X-rays (AUROC, accuracy, sensitivity, specificity, mean P(pneumonia) per class, and a per-image table of practice-image id, true label and P(pneumonia)). Label 1 = PNEUMONIA, 0 = NORMAL.
- Up to **20 practice passes**. For each, you send me an edit file; I apply it once, with no training, to the same practice X-rays and send back the same kind of output, plus how many weights your edit changed. You will never see the images, the features, or anything from the scoring set.

## Rules

- Every change must be one you design and can explain (for example: "the bias puts most normal X-rays above 0.5, so I lower it by …", or "weights above … dominate the logit, so I shrink them by …"). Do not run any optimisation, gradient estimation, finite differences, random search or automated loop over the practice outputs, and do not set weights or the bias by fitting them to the per-image scores.
- Keep a change only if it helps, as you judge from the practice results.
- Say "PRACTICE PASS n" with each file you want run, and "FINAL" with the file to be scored. The scoring set is used exactly once, on the FINAL file.
- With your FINAL file, list every pass you used, what you changed and why.

## How the FINAL edit will be scored

Once, on held-out X-rays from the same collection as the practice X-rays, but of different patients (no child appears in both). Primary metric: AUROC. Also reported: accuracy at P = 0.5, sensitivity (pneumonia recall), specificity (normal recall) and balanced accuracy. It will be compared with the unedited network and with the simplest possible fix (moving the decision threshold to its best point on the same practice X-rays).

## What to send back each time

One Python file, and nothing else in a code block:

```python
# edit_vN.py   (or edit_final.py)
import numpy as np
import math

def edit(weight, bias):
    """weight: numpy array of the 2,048 current weights; bias: float. Return (new_weight, new_bias)."""
    new_weight = weight.copy()
    new_bias = bias
    ...
    return new_weight, new_bias
```

Rules for the file: it may import only `numpy` and `math`; it must not read or write files, use the network, or use random numbers; it must return 2,048 weights and one bias. Your edit receives the current weights as input, so you do not need to copy them out.

Before each file, explain in plain language what you changed and why.

---

[paste PASS 0 output here]
