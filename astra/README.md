# Can an LLM write a working pneumonia classifier without training?

Course-project experiment, after Takehi, Teoh, Zhang & Khattab, *Sparks of Weight-Space
Programming: LLMs can natively read & write neural network weights* (MIT CSAIL, 2026),
https://rikiyat.github.io/blog/weight-space-programming/ — their Exp. 5 (one-shot write, no
feedback) and Exp. 6 (write with a budget of practice forward passes).

An LLM (Astra) writes all 5,569 weights of `WrittenCNN` (`src/cxr/models.py`). Nothing is
fitted to data. The written network is scored once on the official test folder (618 images,
patient-grouped, never shown to the LLM) and compared with:

| Comparison | What it tells us |
|---|---|
| Same architecture, random init, no training | the floor: what "no knowledge" scores |
| Same architecture, trained by gradient descent (A1 protocol, seeds 42 and 43) | the direct answer: written vs trained, same 5,569 slots |
| A1 models (logistic regression, small CNN, ResNet-50, majority) | where the written network sits on the A1 ladder |

## Conditions

| | Mirrors | LLM gets | Feedback | Data the LLM sees |
|---|---|---|---|---|
| **A · zero-shot** | Exp. 5 (MinAtar) | task description, preprocessing code path, network code | none: one reply | none |
| **B · practice** | Exp. 6 (Go) | the same, plus up to 20 practice passes | metrics + per-image scores on practice images | scores and labels of practice (validation) images, never the images; never the test set |

The dataset is deliberately not named in the prompts, so Astra must rely on general medical
and image knowledge rather than anything it may remember about this specific dataset.

## Steps

1. **Condition A.** New Astra chat → paste `PROMPT_A_zero_shot.md` (below its line) → save
   Astra's code as `astra/weights_zeroshot.py` and its full reply as `astra/transcript_A.md`.
2. Check it (no data touched): `python -m cxr.written check --weights astra/weights_zeroshot.py`.
   If it is rejected (wrong shape, forbidden import), send Astra the error text only, and
   note the retry in the transcript.
3. **Condition B.** New Astra chat → `PROMPT_B_practice.md` + prompt A. For each "PRACTICE
   PASS n" file: save as `astra/weights_practice_vN.py`, run
   `python -m cxr.written practice --weights astra/weights_practice_vN.py --rotate --per_image`,
   paste the output back. Save the FINAL file as `astra/weights_practice_final.py` and the chat
   as `astra/transcript_B.md`.
4. **Score once, after both conditions are finished** (so no test result can leak into B):
   ```
   python -m cxr.written evaluate --weights astra/weights_zeroshot.py      --label astra-zeroshot
   python -m cxr.written evaluate --weights astra/weights_practice_final.py --label astra-practice --log astra/practice_log.jsonl
   ```
5. Baselines and table: `bash scripts/run_written.sh` (trains the same architecture with seeds
   42 and 43, about 3 min each on a T4; scores the random-init floor; writes
   `results/written_table.md`).
6. Commit `astra/` (prompts, weights files, transcripts, practice log) and `results/written*`.

## Rules that keep the comparison honest

- The test folder is used once per condition, after all prompting is done.
- No test result is ever shown to Astra.
- Every Astra reply, every weights file and every practice output is kept (`astra/`).
- The weights files are checked automatically: only numpy and math, no file access, no random numbers.

## Log of deviations
- Condition B, pass 1 used the practice images in index order (sorted by class: 199 normal, 57 pneumonia
  in the first slice). From pass 2 on, practice slices come from a fixed shuffled order (seed 0) so every
  slice contains both classes. Nothing else changed; the test folder was not touched.
- Condition B, after Astra declared FINAL (10 practice passes used), the FINAL file was run once more
  through the practice cell (logged as pass 11, practice images only) and that output was shown to Astra.
  Astra made no further change; the FINAL file is byte-for-byte the one it declared. The test folder was
  not touched. Astra's FINAL weights are numerically identical to its passes 4 and 5.
- Astra read the pasted prompts and practice outputs only; the transcript shows no code execution.
- Condition E, design change before any Astra run: the first PASS 0 used our validation set, where the
  trained ResNet-50 is already near-perfect (AUROC 0.9995, specificity 0.97), so the problem the edit is
  meant to fix (test-folder specificity 0.50) was invisible. Practice was moved to a patient-split half of
  the official test folder (306 images, seed 0); scoring uses the other half (312 images, different
  children). The threshold rival uses the same practice half. Astra never saw the first PASS 0.
- Condition E, ResNet-50 seed 42 was retrained in a fresh Colab session (test AUROC 0.966 vs 0.972 in A1;
  GPU training is not bit-for-bit reproducible). Its own result is in results/retrain/.
- Condition E, unlike A-D, Astra ran short scripts (Python/PowerShell) on the attached weight file to compute
  summary numbers for its rules (mean weight, RMS, clip limit, number of weights below the prune cut). These
  read only resnet50_fc.txt and the pasted practice outputs; nothing was fitted to per-image scores. See
  edit/transcript_E.md.

## Condition E result (scoring half, n = 312; 4 of 20 practice passes used)
| Model | Test AUROC [95% CI] | Sensitivity | Specificity | Balanced acc. |
|---|---|---|---|---|
| ResNet-50 trained, unedited | 0.954 [0.925, 0.977] | 0.995 | 0.496 | 0.745 |
| + threshold moved on practice half (Youden, 0.999) | 0.954 [0.925, 0.977] | 0.935 | 0.894 | 0.914 |
| + Astra's edit (centre, prune 170 small weights, bias - ln 2) | 0.954 [0.925, 0.977] | 0.990 | 0.637 | 0.814 |
