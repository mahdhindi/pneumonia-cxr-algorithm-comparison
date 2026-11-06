# Prompt B — write the weights with a budget of practice forward passes

Run only after condition A is finished and its weights file is saved. Start a NEW
Astra conversation (it must not see conversation A). Copy everything below the line,
then paste the WHOLE of PROMPT_A_zero_shot.md after it, from "You will write, by hand"
to the end.

Each time Astra sends a weights file, run one practice pass (cell 3b of the notebook)
and paste back the printed output exactly. Never paste anything from the test set.

---

This is the same task as below, with one difference. You may run up to **20 practice forward passes** before you submit your final weights. A practice pass works like this: you send me a complete weights file; I run your network once, with no training, on 256 practice images of children who are NOT in the scoring set (each pass uses the next 256 of 782 practice images, cycling); I send back AUROC, accuracy, sensitivity, specificity, the mean P(pneumonia) for each class, and a per-image table of practice-image id, true label and your network's P(pneumonia). You will never see the images themselves.

Rules for the practice passes:
- Every change to the weights must be one you design and can explain (for example "conv1 filter 3 responds to the black padding bars, so I set its fc weights to zero"). Do not run any optimisation, gradient estimation, finite differences, random search or automated loop over the practice outputs, and do not set weights by fitting to the per-image scores.
- Keep a change only if it helps, as you judge from the practice results.
- Say "PRACTICE PASS n" with each file you want run, and "FINAL" with the file to be scored. The scoring set is used exactly once, on the FINAL file.
- With your FINAL file, list every pass you used and what you changed and why.
