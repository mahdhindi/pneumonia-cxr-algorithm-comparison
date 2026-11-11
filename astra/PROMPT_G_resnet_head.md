# Prompt G — write the final layer of a frozen ImageNet ResNet-50, with practice passes + feature report

Condition G = condition F's approach (practice passes with a per-unit report), applied to a bigger,
pretrained feature extractor. Astra writes only the final layer: 2,048 weights and one bias. The
backbone is a standard ImageNet ResNet-50, trained on everyday photographs and never on X-rays, and it
stays frozen. No example images are attached: the 2,048 features have no names, so pictures cannot
tell Astra which feature is which (condition D showed images add little on top of practice anyway).

How to run it:
1. Colab: cell G1 (once per session) computes the frozen features and prints PASS 0 (not counted).
2. Start a NEW Astra chat from an empty folder, with no attachments. Paste everything below the line,
   then paste the whole PASS 0 output at the end of the same message (or attach it as a .txt file).
3. Each "PRACTICE PASS n" file: save it with %%writefile as astra/head/head_vN.py, run cell G2 with
   N = n, and paste back everything it prints (or attach it as a .txt file; it is long).
4. At FINAL: save astra/head/head_final.py and send the file and the conversation for checking
   before running cell G3.

---

You will write, by hand, the final layer of a neural network that detects pneumonia in frontal chest X-rays of children. No training of any kind will happen: no loss function, no gradients, no optimisation algorithm, no tools or code execution, and no fitting of numbers to data. Your layer will be loaded exactly as you write it and scored once on held-out X-rays of children you will never see.

## The network

A ResNet-50 pretrained on ImageNet (everyday photographs: animals, objects, scenes). It has **never been trained on X-rays**, and its body is frozen: nothing in it will change. For each X-ray it produces 2,048 features (each a number ≥ 0, after ReLU and global average pooling). The features have no names; what each one responds to in an X-ray is unknown. You write the only remaining layer:

```
logit = sum over i of weight[i] * feature[i]  +  bias          (i = 0 … 2047)
P(pneumonia) = sigmoid(logit)
```

How every image is prepared: converted to 8-bit grayscale, padded to a square with black bars, resized to 224 × 224, copied into the three colour channels and normalised with the standard ImageNet mean and standard deviation. No augmentation at scoring time.

## What you get

- **PASS 0** (pasted below this prompt, free, not counted): the feature report for the first 256 practice images, measured with an all-zero layer, so every P is 0.5 and the AUROC line means nothing yet.
- Up to **20 practice passes**. For each, you send me a layer file. I run it once, with no training, on the next 256 practice images (children who are NOT in the scoring set; there are 782 practice images and the passes cycle through them). I send back:
  1. AUROC, accuracy, sensitivity, specificity, and the mean P(pneumonia) for each class;
  2. a per-image table of practice-image id, true label and your P(pneumonia);
  3. the **FEATURE REPORT**: for each of the 2,048 features, its mean value on normal and on pneumonia images, its *separation* (the AUROC of that feature on its own: 0.5 = it does not tell the classes apart; above 0.5 = higher on pneumonia; below 0.5 = higher on normal), the share of images on which it is *silent* (zero), and its contribution to your logit (your weight × its mean value) for each class. The rows are sorted by separation, highest first, with a one-line summary at the top.

You will never see the practice images themselves.

## Rules

- Every weight and the bias must be a choice you design and can explain (for example: "features f123, f456 and f789 are consistently higher on pneumonia across passes, so I give each weight +w", or "the bias puts most normal images above 0.5, so I lower it by …"). You may choose which features to use and give them hand-picked values.
- Do not run any optimisation, gradient estimation, finite differences, random search, code or automated loop over the practice outputs. Read them and reason.
- Do not compute weights as a formula of the reported numbers or the per-image scores (for example, setting each weight proportional to its separation, or to the difference of class means, is not allowed). Use the report to decide *what* to use; set the values by design.
- Keep a change only if it helps, as you judge from the practice results.
- Say "PRACTICE PASS n" with each file you want run, and "FINAL" with the file to be scored. The scoring set is used exactly once, on the FINAL file.
- With your FINAL file, list every pass you used, what you changed and why.

## How your layer will be scored

Once, on held-out X-rays. Primary metric: AUROC. Also reported: accuracy at P = 0.5, sensitivity (pneumonia recall) and specificity (normal recall). It will be compared with layers obtained in other ways on the same frozen features, and with networks trained end to end.

## What to send back each time

One Python file, and nothing else in a code block:

```python
# head_vN.py   (or head_final.py)
import numpy as np
import math

def make_head():
    """Return (weight, bias): weight = 2,048 numbers for features 0..2047, bias = one float."""
    weight = np.zeros(2048)
    ...
    bias = 0.0
    return weight, bias
```

Rules for the file: it may import only `numpy` and `math`; it must not read or write files or use random numbers; every value must follow from constants or formulas you chose (a list of feature indices you picked is fine).

Before each file, explain in plain language what you changed and why. Before the code of your FINAL file, also give your own prediction of the AUROC you expect.
