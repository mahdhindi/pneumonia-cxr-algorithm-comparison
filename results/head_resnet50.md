# Condition G: linear head on a frozen ImageNet ResNet-50 (official test folder, n = 618)

| Head on frozen ImageNet ResNet-50 | How | Test AUROC [95% CI] | Sens. | Spec. | Features used | Size-matched AUROC (140 img) |
|---|---|---|---|---|---|---|
| Head written by Astra | no training; 17 practice passes + feature report | 0.902 [0.877, 0.922] | 0.935 | 0.589 | 26 | 0.724 |
| Selection rule (mechanical rival) | top-32 features by separation on the 782 practice images seen; no gradients | 0.883 [0.853, 0.905] | 0.817 | 0.801 | 32 | 0.669 |
| Linear probe (trained) | logistic regression on 4,424 training images, C=0.01 chosen on practice; fit 0.9 s | 0.947 [0.926, 0.963] | 0.992 | 0.532 | 2047 | 0.862 |
| Best of 17 random heads | matched random search, 100 repetitions: median [5th-95th pct]; max 0.800; share >= Astra 0% | 0.678 [0.573, 0.781] | | | 2048 | 0.571 (median) |
