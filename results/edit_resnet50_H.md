# Condition H: LLM edit of the trained ResNet-50's final layer (scoring half of the official test folder (patients not in the practice half), n = 312; size-matched pairs within it: 72 images)

| Model | How | Test AUROC [95% CI] | Accuracy | Sensitivity | Specificity | Balanced acc. | Size-matched AUROC |
|---|---|---|---|---|---|---|---|
| ResNet-50, trained (unedited) | gradient descent, A1 protocol, seed 42 | 0.954 [0.925, 0.977] | 0.814 | 0.995 | 0.496 | 0.745 | 0.879 |
| ResNet-50 + threshold moved on practice set | rival: one number changed (threshold 0.999, Youden on the 306 practice images) | 0.954 [0.925, 0.977] | 0.920 | 0.935 | 0.894 | 0.914 | 0.879 |
| ResNet-50 edited by Astra | no gradients; 2048 of 2,048 weights changed, bias +0.018 -> -2.000 | 0.976 [0.959, 0.990] | 0.923 | 0.975 | 0.832 | 0.903 | 0.922 |
| Last layer re-trained on the practice half | rival: logistic regression on the same 306 practice images (C=0.1 by 5-fold CV); gradient-based | 0.985 [0.974, 0.995] | 0.949 | 0.970 | 0.912 | 0.941 | 0.925 |
