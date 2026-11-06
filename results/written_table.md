# LLM-written weights vs trained weights (patient-grouped split, official test folder, n = 618)

| Model | How the weights were obtained | Runs | Test AUROC | Accuracy | Sensitivity | Specificity | Params | Train (min) |
|---|---|---|---|---|---|---|---|---|
| WrittenCNN, weights written by Astra (astra-practice) | no training; 10 practice passes; AUROC 95% CI [0.695, 0.779] | 1 | 0.737 | 0.673 | 0.587 | 0.818 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra (astra-zeroshot) | no training; 0 practice passes; AUROC 95% CI [0.473, 0.565] | 1 | 0.519 | 0.549 | 0.742 | 0.225 | 5,569 | 0.0 |
| WrittenCNN, random init | no training (floor) | 2 | 0.538 ± 0.197 | 0.661 | 0.872 | 0.307 | 5,569 | 0.0 |
| WrittenCNN, trained | gradient descent, A1 protocol | 2 | 0.882 ± 0.012 | 0.790 | 0.970 | 0.487 | 5,569 | 3.3 |
| Logistic regression (pixels) | A1 result | 2 | 0.895 ± 0.004 | 0.774 | 0.982 | 0.426 | 150,529 | 3.0 |
| Small CNN (scratch) | A1 result | 2 | 0.942 ± 0.004 | 0.826 | 0.996 | 0.541 | 389,153 | 3.4 |
| ResNet-50 (ImageNet) | A1 result | 2 | 0.971 ± 0.001 | 0.862 | 0.997 | 0.636 | 23,510,081 | 4.4 |
| Majority class | A1 result | 2 | 0.500 ± 0.000 | 0.626 | 1.000 | 0.000 | 0 | 0.0 |
| Random untrained networks (no feedback) | 200 draws; median [5th-95th pct]; share >= 0.519: 48%, 0.737: 9%, 0.882: 0% | 200 | 0.491 [0.245, 0.752] | | | | 5,569 | 0.0 |
| Best of 10 random networks, chosen on practice slices (Astra's budget) | 100 draws; median [5th-95th pct]; share >= 0.519: 97%, 0.737: 28%, 0.882: 0% | 100 | 0.716 [0.573, 0.786] | | | | 5,569 | 0.0 |
