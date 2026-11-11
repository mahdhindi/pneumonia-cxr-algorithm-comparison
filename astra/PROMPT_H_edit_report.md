# Prompt H — edit the final layer of our TRAINED ResNet-50, with practice passes + per-feature report

Condition H = condition E (edit our fine-tuned ResNet-50's last layer) with condition G's feedback (a
per-feature report after every pass). Practice = one patient-split half of the official test folder
(306 images); scoring = the other half (312 images, different children). The validation set is not
used: the trained model is already near-perfect there (AUROC 0.9995), so nothing to fix shows.

How to run it:
1. Colab: cell H1 (once per session). It trains/loads the ResNet-50, extracts features, and downloads
   `resnet50_fc.txt` (the current 2,048 weights + bias) and `pass_00_output.txt` (PASS 0, free).
2. Start a NEW Astra chat from an empty folder. Attach both files. Paste everything below the line.
3. Each "PRACTICE PASS n" file: save it with %%writefile as astra/edit_h/edit_vN.py, run cell H2 with
   N = n, and attach the downloaded pass_NN_output.txt.
4. At FINAL: save astra/edit_h/edit_final.py and send the file and the conversation for checking
   before running cell H3.

---

Below is a trained neural network that detects pneumonia in frontal chest X-rays of children. It was trained normally (gradient descent) and it works, but not as well as it could on X-rays from a different collection. Your job is to improve it by editing its final layer by hand. No training of any kind will happen: no loss function, no gradients, no optimisation algorithm, and no fitting of numbers to data.

## The network

A ResNet-50 (pretrained on everyday photographs, then fine-tuned on chest X-rays). For each X-ray, its body produces 2,048 features (each a number ≥ 0, after ReLU and global average pooling). Its final layer turns them into a prediction:

    logit = sum over i of weight[i] * feature[i]  +  bias          (i = 0 … 2047)
    P(pneumonia) = sigmoid(logit)

You may change only this final layer: the 2,048 weights and the bias. The body is frozen. The attached file `resnet50_fc.txt` lists every current weight (index, value) and the bias. The features have no names; what each one responds to is unknown.

## What you get

- Attached: **PASS 0** (`pass_00_output.txt`, free, not counted): the unedited network on all 306 practice X-rays.
- Up to **20 practice passes**. For each, you send me an edit file; I apply it once, with no training, to the same 306 practice X-rays (children who are not in the scoring set) and send back the same kind of output as a file. Each output contains:
  1. AUROC, accuracy, sensitivity, specificity, the mean P(pneumonia) per class, and how many weights your edit changed;
  2. a per-image table of practice-image id, true label and P(pneumonia);
  3. the **FEATURE REPORT**: for each of the 2,048 features, its mean value on normal and on pneumonia images, its *separation* (the AUROC of that feature on its own: 0.5 = it does not tell the classes apart; above 0.5 = higher on pneumonia; below 0.5 = higher on normal), the share of images on which it is *silent* (zero), and its current contribution to the logit (weight × mean value) for each class. Rows are sorted by separation, highest first, with a one-line summary at the top. Comparing a feature's separation with the sign and size of its current weight shows where the trained layer and these X-rays disagree.

You will never see the images, or anything from the scoring set.

## Tools

You may use a tool **only to read** the attached files (for example, to display their lines). Do not run code or any computation, search, sorting or analysis over them: read and reason.

## Rules

- Every change must be one you design and can explain (for example: "f123 now separates in the opposite direction to its weight, so I set its weight to 0", "these five features consistently separate on these X-rays, so I add +w to each", or "the bias puts most normal X-rays above 0.5, so I lower it by …").
- Do not run any optimisation, gradient estimation, finite differences, random search or automated loop over the practice outputs. Do not compute weights as a formula of the reported numbers or the per-image scores (for example, setting each weight proportional to its separation is not allowed). Use the report to decide *what* to change; set the values by design.
- Keep a change only if it helps, as you judge from the practice results.
- Say "PRACTICE PASS n" with each file you want run, and "FINAL" with the file to be scored. The scoring set is used exactly once, on the FINAL file.
- With your FINAL file, list every pass you used, what you changed and why, and give your own prediction of the AUROC you expect.

## How the FINAL edit will be scored

Once, on 312 held-out X-rays from the same collection as the practice X-rays, but of different children. Primary metric: AUROC. Also reported: accuracy at P = 0.5, sensitivity, specificity and balanced accuracy. It will be compared with the unedited network, with moving its decision threshold, and with a final layer re-trained on the same practice X-rays.

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

Rules for the file: it may import only `numpy` and `math`; it must not read or write files or use random numbers; it must return 2,048 weights and one bias. Your edit receives the trained weights as input, so you do not need to copy them out. Every value must follow from constants or formulas you chose (a list of feature indices you picked is fine).

Before each file, explain in plain language what you changed and why.
