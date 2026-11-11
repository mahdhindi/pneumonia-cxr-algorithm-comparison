# LLM-written weights vs trained weights (patient-grouped split, official test folder, n = 618)

| Model | How the weights were obtained | Runs | Test AUROC | Accuracy | Sensitivity | Specificity | Params | Train (min) |
|---|---|---|---|---|---|---|---|---|
| WrittenCNN, weights written by Astra (astra-diagnose-r1) | no training; 4 practice passes; AUROC 95% CI [0.848, 0.900] | 1 | 0.876 | 0.751 | 0.961 | 0.398 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra (astra-diagnose-r2) | no training; 4 practice passes; AUROC 95% CI [0.835, 0.891] | 1 | 0.865 | 0.773 | 0.886 | 0.584 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra (astra-examples-practice) | no training; 11 practice passes; AUROC 95% CI [0.705, 0.785] | 1 | 0.745 | 0.702 | 0.693 | 0.719 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra (astra-examples) | no training; 0 practice passes; AUROC 95% CI [0.553, 0.646] | 1 | 0.599 | 0.610 | 0.889 | 0.143 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra (astra-practice) | no training; 10 practice passes; AUROC 95% CI [0.695, 0.779] | 1 | 0.737 | 0.673 | 0.587 | 0.818 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra (astra-zeroshot) | no training; 0 practice passes; AUROC 95% CI [0.473, 0.565] | 1 | 0.519 | 0.549 | 0.742 | 0.225 | 5,569 | 0.0 |
| WrittenCNN, weights written by Astra, condition F (mean of 2 independent chats) | no training; images + practice + channel report | 2 | 0.870 ± 0.005 | 0.762 | 0.924 | 0.491 | 5,569 | 0.0 |
| WrittenCNN, random init | no training (floor) | 2 | 0.538 ± 0.197 | 0.661 | 0.872 | 0.307 | 5,569 | 0.0 |
| WrittenCNN, trained | gradient descent, A1 protocol | 2 | 0.882 ± 0.012 | 0.790 | 0.970 | 0.487 | 5,569 | 3.3 |
| Logistic regression (pixels) | A1 result | 2 | 0.895 ± 0.004 | 0.774 | 0.982 | 0.426 | 150,529 | 3.0 |
| Small CNN (scratch) | A1 result | 2 | 0.942 ± 0.004 | 0.826 | 0.996 | 0.541 | 389,153 | 3.4 |
| ResNet-50 (ImageNet) | A1 result | 2 | 0.971 ± 0.001 | 0.862 | 0.997 | 0.636 | 23,510,081 | 4.4 |
| Majority class | A1 result | 2 | 0.500 ± 0.000 | 0.626 | 1.000 | 0.000 | 0 | 0.0 |
| Random untrained networks (no feedback) | 200 draws; median [5th-95th pct]; share >= 0.519: 48%, 0.737: 9%, 0.882: 0% | 200 | 0.491 [0.245, 0.752] | | | | 5,569 | 0.0 |
| Best of 4 random networks, chosen on practice slices (matched budget) | 100 draws; median [5th-95th pct]; share >= 0.519: 88%, 0.737: 22%, 0.882: 0% | 100 | 0.691 [0.432, 0.769] | | | | 5,569 | 0.0 |
| Best of 10 random networks, chosen on practice slices (matched budget) | 100 draws; median [5th-95th pct]; share >= 0.519: 97%, 0.737: 28%, 0.882: 0% | 100 | 0.716 [0.573, 0.786] | | | | 5,569 | 0.0 |
| Best of 11 random networks, chosen on practice slices (matched budget) | 100 draws; median [5th-95th pct]; share >= 0.519: 100%, 0.737: 34%, 0.882: 0% | 100 | 0.719 [0.632, 0.790] | | | | 5,569 | 0.0 |
