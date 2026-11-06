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
