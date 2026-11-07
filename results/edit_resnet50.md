# Condition E: LLM edit of the trained ResNet-50's final layer (scoring half of the official test folder (patients not in the practice half), n = 312)

| Model | How | Test AUROC [95% CI] | Accuracy | Sensitivity | Specificity | Balanced acc. |
|---|---|---|---|---|---|---|
| ResNet-50, trained (unedited) | gradient descent, A1 protocol, seed 42 | 0.954 [0.925, 0.977] | 0.814 | 0.995 | 0.496 | 0.745 |
| ResNet-50 + threshold moved on practice set | rival: one number changed (threshold 0.999, Youden on the 306 practice images) | 0.954 [0.925, 0.977] | 0.920 | 0.935 | 0.894 | 0.914 |
| ResNet-50 edited by Astra | no gradients; 2048 of 2,048 weights changed, bias +0.018 -> -0.675 | 0.954 [0.925, 0.977] | 0.862 | 0.990 | 0.637 | 0.814 |
