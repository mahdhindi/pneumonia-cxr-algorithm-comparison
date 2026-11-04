# Assignment 2 reality check - Singh et al. (2024) vs this pipeline

| Model | Setup | LR | Seeds | AUROC | Accuracy | Sensitivity | Specificity | Balanced acc. | Params (M) | Train (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT DeiT-B/16 (ImageNet, fine-tuned) | Reported by Singh et al. (2024), test n=586, 1 run | 1e-05 | 1 | 0.966 | 0.976 | 0.949 | 0.981 | n/a | 85.8 | not reported |
| ResNet-50 (ImageNet, fine-tuned) | Reported by Singh et al. (2024), test n=586, 1 run | n/a | 1 | not reported | 0.824 | not reported | not reported | n/a | head only | not reported |
| DenseNet-121 (ImageNet, fine-tuned) | Reported by Singh et al. (2024), test n=586, 1 run | n/a | 1 | not reported | 0.918 | not reported | not reported | n/a | head only | not reported |
| Majority class | A1 protocol: patient-grouped split, official test folder (n=618) | 0 | 2 | 0.500 ± 0.000 | 0.626 ± 0.000 | 1.000 | 0.000 | 0.500 | 0.0 | 0.0 |
| Logistic regression (pixels) | A1 protocol: patient-grouped split, official test folder (n=618) | 0.0001 | 2 | 0.895 ± 0.006 | 0.774 ± 0.042 | 0.982 | 0.426 | 0.704 | 0.2 | 3.0 |
| Small CNN (scratch) | A1 protocol: patient-grouped split, official test folder (n=618) | 0.001 | 2 | 0.942 ± 0.006 | 0.826 ± 0.003 | 0.996 | 0.541 | 0.769 | 0.4 | 3.4 |
| ResNet-50 (ImageNet, fine-tuned) | A1 protocol: patient-grouped split, official test folder (n=618) | 0.0001 | 2 | 0.971 ± 0.001 | 0.862 ± 0.034 | 0.997 | 0.636 | 0.817 | 23.5 | 4.4 |
| ResNet-50 (ImageNet, fine-tuned) | Singh et al. protocol: random 80/10/10 by image | 0.0001 | 2 | 0.995 ± 0.004 | 0.972 ± 0.004 | 0.969 | 0.978 | 0.974 | 23.5 | 5.0 |
| DenseNet-121 (ImageNet, fine-tuned) | A1 protocol: patient-grouped split, official test folder (n=618) | 0.0001 | 2 | 0.936 ± 0.017 | 0.822 ± 0.027 | 0.999 | 0.526 | 0.762 | 7.0 | 5.9 |
| ViT DeiT-B/16 (ImageNet, fine-tuned) | A1 protocol: patient-grouped split, official test folder (n=618) | 1e-05 | 2 | 0.973 ± 0.001 | 0.824 ± 0.027 | 0.999 | 0.530 | 0.765 | 85.8 | 9.2 |
| ViT DeiT-B/16 (ImageNet, fine-tuned) | Singh et al. protocol: random 80/10/10 by image | 1e-05 | 2 | 0.997 ± 0.002 | 0.980 ± 0.006 | 0.984 | 0.972 | 0.978 | 85.8 | 9.0 |
