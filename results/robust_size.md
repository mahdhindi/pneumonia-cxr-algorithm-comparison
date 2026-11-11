# Robustness check: size-matched test subset (70 normal/pneumonia pairs, original width and height within ~5 %; 140 of 618 test images)

| Model | How | Full test AUROC (n=618) | Size-matched AUROC [95% CI] | Drop | Matched, wider caliper (82 pairs) | Spearman(logit, log height) normal / pneumonia |
|---|---|---|---|---|---|---|
| Original image height alone (smaller = pneumonia) | acquisition shortcut, reference | 0.921 | 0.510 | +0.412 | 0.513 | – |
| Aspect ratio alone (wider = pneumonia) | acquisition shortcut, reference | 0.703 | 0.500 | +0.203 | 0.463 | – |
| Astra, practice (B) | LLM-written, no training | 0.737 | 0.677 [0.583, 0.770] | +0.060 | 0.683 | -0.31 / -0.09 |
| Astra, images + practice (D) | LLM-written, no training | 0.745 | 0.693 [0.604, 0.787] | +0.051 | 0.697 | -0.22 / -0.11 |
| Astra, images + practice + channel report (F, run 1) | LLM-written, no training | 0.876 | 0.761 [0.671, 0.835] | +0.114 | 0.765 | -0.16 / -0.51 |
| Same CNN, trained (seed 42) | gradient descent, A1 protocol | 0.894 | 0.758 [0.674, 0.837] | +0.136 | 0.750 | -0.71 / -0.59 |
| Same CNN, trained (seed 43) | gradient descent, A1 protocol | 0.870 | 0.726 [0.638, 0.811] | +0.144 | 0.714 | -0.74 / -0.59 |
| ResNet-50, trained (seed 42) | gradient descent, A1 protocol | 0.966 | 0.905 [0.848, 0.950] | +0.061 | 0.911 | -0.59 / +0.03 |
