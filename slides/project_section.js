// ============================================================== P1 divider
pres.addSection({ title: "Project" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Project" });
s.addText("PART 2 · COURSE PROJECT (OPTION 3: AI AGENTS)", { placeholder: "kicker" });
s.addText("Can an LLM write a pneumonia classifier's weights, with no training at all?", { placeholder: "title" });
s.addText("Part 1 showed that the evaluation decides what a number means. Part 2 keeps our fair pipeline fixed and changes who sets the weights.", { placeholder: "sub" });
card(s, 0.7, 4.2, 5.8, 2.5, "Where it came from", "Our critique's next steps were an external test set and a training-size sweep. We took Option 3 instead: in class we met MIT's \"Sparks of Weight-Space Programming\" (Takehi et al., Nov 2026, blog post), where GPT-6 Astra wrote and edited network weights directly. We have access to Astra.", { fill: "0B4F5F", headColor: C.accent4, color: "FFFFFF", size: 13 });
card(s, 6.85, 4.2, 5.8, 2.5, "Research question (one sentence)", "Can GPT-6 Astra, writing all 5,569 weights of a small CNN by reasoning alone, beat a random search given the same number of practice evaluations on test AUROC, and how close does it get to the same CNN trained by gradient descent?", { fill: "7A2E1C", headColor: C.accent6, color: "FFFFFF", size: 13.5 });
notes(s, "Mahd (0.5 min)", "Part two. Our critique pointed to an external test set, but in class we met MIT's weight-space programming work, where the same LLM we have access to, GPT-6 Astra, wrote network weights directly. Our question names the method, the problem, the fair baseline and the metric: can Astra beat a matched random search, and how close does it get to training?");

// ============================================================== P2 MIT modes -> our conditions
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("What MIT did, and what we copied", { placeholder: "title" });
s.addText("Their wins came from tasks with rules the LLM could read. Pneumonia has no source code: only labelled X-rays define it.", { placeholder: "lead" });
s.addTable([
  [hdr("MIT mode"), hdr("What the LLM got"), hdr("Their result"), hdr("Our version")],
  [td("Write from scratch", { align: "left", bold: true, color: X.accent2 }), td("Game rules, source code, network shape; no data", { align: "left" }), td("MinAtar policies beat PPO at 100 M steps", { align: "left" }), td("A–D, F: Astra writes all 5,569 weights of a small CNN · G: Astra writes the last layer of an ImageNet ResNet-50", { align: "left", bold: true, color: X.accent2 })],
  [td("Edit", { align: "left", bold: true }), td("A failing trained network + forward passes", { align: "left" }), td("Qwen MATH-500: 7.9 → 48.1 %", { align: "left" }), td("E: Astra edits the last layer of our trained ResNet-50", { align: "left" })],
  [td("Read", { align: "left", bold: true }), td("Raw weights: \"what was this trained on?\"", { align: "left" }), td("Named the game or language", { align: "left" }), td("Not used", { color: X.accent5 })],
], { x: 0.6, y: 1.9, w: 12.15, colW: [1.9, 3.3, 2.9, 4.05], rowH: 0.62, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
card(s, 0.6, 4.6, 5.95, 2.2, "Why this is a hard test", "In the games the LLM effectively compiled known rules into weights. Here, whatever it writes must come from what it knows about radiographs and from the feedback we give it.", { size: 13.5 });
card(s, 6.8, 4.6, 5.95, 2.2, "What makes it fair", "Same data, patient-grouped splits, preprocessing and seeds as Assignment 1. Same network for every row. Each rival gets exactly as many practice evaluations as Astra used. The test set is scored once per condition.", { size: 13.5 });
notes(s, "Mahd (0.7 min)", "MIT had three modes. We copied two: writing weights from scratch, and editing a trained network. The difference that matters: their games came with source code; pneumonia is defined only by labelled images. Fairness comes from Assignment 1: same splits, same preprocessing, same network for every row, and every rival gets the same number of practice evaluations as Astra.");

// ============================================================== P3 setup + conditions
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Setup: one small CNN, five things Astra could be given", { placeholder: "title" });
s.addText("Each condition is a fresh Astra chat. It returns a Python file that builds every weight from constants; no training of any kind.", { placeholder: "lead" });
const blocks = [["X-ray", "1 × 224 × 224"], ["conv1", "16 filters 5×5"], ["conv2", "32 channels 3×3"], ["4 × 4 grid", "512 features"], ["fc", "512 → 1"], ["P(pneumonia)", "5,569 weights"]];
blocks.forEach(([h, b], i) => {
  const x = 0.6 + i * 2.05;
  box(s, x, 1.85, 1.8, 0.95, i === 0 || i === 5 ? X.lt2 : X.accent4);
  txt(s, h, x + 0.1, 1.92, 1.6, 0.35, { fontSize: 13, bold: true, color: C.text2, align: "center" });
  txt(s, b, x + 0.1, 2.3, 1.6, 0.4, { fontSize: 11, color: C.text1, align: "center" });
  if (i < 5) s.addShape(pres.ShapeType.line, { x: x + 1.82, y: 2.32, w: 0.22, h: 0, line: { color: X.accent3, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
});
s.addTable([
  [hdr("Condition"), hdr("What Astra gets"), hdr("Practice evaluations")],
  [td("A", { bold: true }), td("The task, the preprocessing and the network code. Nothing else.", { align: "left" }), td("0 (one reply)")],
  [td("C", { bold: true }), td("+ 16 labelled example X-rays to look at", { align: "left" }), td("0 (one reply)")],
  [td("B", { bold: true }), td("+ practice passes: we run its file on 256 practice images and return scores", { align: "left" }), td("up to 20")],
  [td("D", { bold: true }), td("+ images and practice passes", { align: "left" }), td("up to 20")],
  [td("F", { bold: true, color: X.accent2 }), td("+ images, practice and a channel report (how every filter responds to each class) · 3 independent chats", { align: "left", color: X.accent2, bold: true }), td("up to 20")],
], { x: 0.6, y: 3.0, w: 8.0, colW: [1.1, 5.2, 1.7], rowH: 0.48, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
card(s, 8.85, 3.0, 3.9, 3.85, "Rules (checked by code)", "No gradients, no fitting, no search loops: every change is designed and explained.\nOnly numpy and math, no random numbers, no file access: a static check and an isolated process enforce it.\nDataset not named, no image sizes shown.\nTest set scored once.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 12.5 });
notes(s, "Abdulla (1 min)", "The network is small on purpose: 5,569 weights, so an LLM can write all of them. The five conditions are an ablation of information: nothing; example images; practice scores; both; and in F, a channel report that tells Astra how each of its own filters responds to normal and pneumonia images. F was run in three separate chats. The rules on the right are enforced by code, not trust.");

// ============================================================== P4 practice loop + rivals
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("The practice loop, and the rivals that make it a fair fight", { placeholder: "title" });
s.addText("Astra never sees a practice image, only numbers about how its own network did.", { placeholder: "lead" });
const loop = [["Astra writes", "a weights file\n(by reasoning)"], ["We check it", "static rules +\nisolated process"], ["One forward pass", "next 256 practice\nimages, no training"], ["We return", "AUROC, sens., spec.,\nper-image scores\n+ channel report (F)"]];
loop.forEach(([h, b], i) => {
  const x = 0.6 + i * 3.1;
  box(s, x, 1.9, 2.75, 1.55, i === 3 ? X.accent4 : X.lt2);
  txt(s, h, x + 0.12, 2.0, 2.5, 0.35, { fontSize: 14, bold: true, color: C.text2, align: "center" });
  txt(s, b, x + 0.12, 2.4, 2.5, 1.0, { fontSize: 12, color: C.text1, align: "center" });
  if (i < 3) s.addShape(pres.ShapeType.line, { x: x + 2.78, y: 2.67, w: 0.3, h: 0, line: { color: X.accent3, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
});
txt(s, "↺  Astra reads the result, decides one change, and repeats; it says FINAL when done (it stopped after 4 passes in every F chat).", 0.6, 3.55, 12.1, 0.4, { fontSize: 13, color: C.accent5 });
card(s, 0.6, 4.15, 3.9, 2.7, "A real channel-report line (F, run 2)", "conv2 channel 8 (local intensity range): separation 0.061, i.e. clearly higher on NORMAL images.\nAstra's next file read only this channel, with weight −1: practice AUROC 0.59 → 0.93.", { size: 13 });
card(s, 4.72, 4.15, 3.9, 2.7, "Rival 1 · matched random search", "Draw N random networks, score each on the same practice slices Astra saw, keep the best; N = Astra's number of passes. Repeated 100 times.", { size: 13 });
card(s, 8.85, 4.15, 3.9, 2.7, "Rival 2 · gradient training", "The same 5,569-weight CNN trained with the Assignment 1 recipe (AdamW, 12 epochs, seeds 42 and 43): 3.3 min on a T4.", { size: 13 });
notes(s, "Abdulla (0.8 min)", "This is one practice pass. Astra writes a file, we check it, run it once on 256 practice images, and send back the scores. In condition F we also send the channel report. Here is a real line from run 2: one of Astra's own channels was much higher on normal images; Astra rebuilt its output around it and practice AUROC jumped from 0.59 to 0.93. Two rivals: random search with exactly the same number of tries, and the same network trained normally.");

// ============================================================== P5 main result (figure)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Result: feedback makes it work, and the channel report brings it close to training", { placeholder: "title" });
s.addText("Test AUROC on the 618 official test images (patient-grouped). Bars: 95 % CI, or 5th–95th percentile for random rows.", { placeholder: "lead" });
s.addImage({ path: "/home/claude/a2repo/results/figures/project_ladder.png", x: 0.6, y: 1.8, w: 6.53, h: 5.05 });
stat(s, 7.6, 1.85, 2.5, "0.519", "nothing given (A): chance", C.accent2);
stat(s, 10.2, 1.85, 2.5, "0.737", "practice scores (B); images add only +0.008 (D)", C.accent2);
stat(s, 7.6, 3.55, 2.5, "0.859", "± 0.016, channel report (F), 3 chats", C.accent1);
stat(s, 10.2, 3.55, 2.5, "0.882", "± 0.012, same CNN trained by gradient descent", C.accent5);
card(s, 7.6, 5.25, 5.15, 1.6, "Read it as", "Every F chat beat all 100 matched random searches (best 0.821). With practice alone (B), Astra was not clearly better than random search (28 % of searches matched it).", { size: 12.5 });
notes(s, "Khaled (1.3 min)", "The whole ladder on one chart. With nothing, Astra is at chance: 0.519. Looking at 16 images barely helps. Practice scores are what make it work: 0.737, but at that point random search with the same feedback matches it a quarter of the time. The channel report changes the picture: three independent chats reach 0.876, 0.865 and 0.838, mean 0.859, and every one beats all hundred random searches. The same network trained by gradient descent gets 0.882, so the gap is 0.02.");

// ============================================================== P6 what Astra wrote
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Three independent chats, one rule: less fine texture means pneumonia", { placeholder: "title" });
s.addText("Each chat started from a different weak design; the first channel report turned each one around.", { placeholder: "lead" });
s.addChart(pres.ChartType.line, [
  { name: "F run 1", labels: ["Pass 1", "Pass 2", "Pass 3", "Pass 4"], values: [0.631, 0.893, 0.942, 0.931] },
  { name: "F run 2", labels: ["Pass 1", "Pass 2", "Pass 3", "Pass 4"], values: [0.589, 0.933, 0.918, 0.938] },
  { name: "F run 3", labels: ["Pass 1", "Pass 2", "Pass 3", "Pass 4"], values: [0.613, 0.933, 0.938, 0.918] },
], { x: 0.6, y: 1.8, w: 6.6, h: 5.05, chartColors: [X.accent1, X.accent2, X.dk2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 8,
  valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, showLegend: true, legendPos: "b", showTitle: true, title: "Practice AUROC per pass (each on a new 256-image slice)", ...chartText });
card(s, 7.5, 1.8, 5.25, 1.6, "What changed after pass 1", "Astra saw that its own texture channels (edge strength, local contrast) were consistently higher on NORMAL images, and rebuilt its output around them with round weights (−1, −0.5).", { size: 12.5 });
card(s, 7.5, 3.5, 5.25, 1.5, "How simple the final rule is", "Run 3's final network reads one channel, average edge strength, with weight −1 per grid cell and bias +1.25. Healthy lungs show more fine structure.", { size: 12.5 });
card(s, 7.5, 5.1, 5.25, 1.75, "Honest framing", "No weight was fitted, but the report is label-informed: it tells Astra which of its own hand-built filters separate the classes. That is feature selection by reasoning: no training, but not data-free.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 12.5 });
notes(s, "Khaled (1 min)", "What did Astra actually write? Each chat began with a brightness-and-opacity design scoring about 0.6. After the first channel report, all three noticed independently that their texture channels were higher on healthy lungs, and rebuilt the output around them. Run 3 ended up reading a single channel. We are careful about what this means: no weight was fitted, but the report uses labels, so this is feature selection by reasoning, not learning from nothing.");

// ============================================================== P7 condition G
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Stronger features (G): Astra writes the last layer of a frozen ImageNet ResNet-50", { placeholder: "title" });
s.addText("Backbone trained only on everyday photos, never on X-rays. Astra writes 2,048 weights + 1 bias; same practice loop, one report row per feature.", { placeholder: "lead" });
s.addTable([
  [hdr("Last layer obtained by"), hdr("Test AUROC [95 % CI]"), hdr("Features used"), hdr("Size-matched AUROC")],
  [td("Astra · 17 practice passes + feature report", { align: "left", bold: true, color: X.accent2 }), td("0.902 [0.877, 0.922]", { bold: true, color: X.accent2 }), td("26"), td("0.724")],
  [td("Selection rule: top features by separation (no LLM, no gradients)", { align: "left" }), td("0.883 [0.853, 0.905]"), td("32"), td("0.669")],
  [td("Best of 17 random layers (100 repeats, median)", { align: "left" }), td("0.678 (max 0.800)"), td("2,048"), td("0.571")],
  [td("Linear probe, trained (logistic regression)", { align: "left", bold: true, color: X.accent1 }), td("0.947 [0.926, 0.963]", { bold: true, color: X.accent1 }), td("2,047"), td("0.862")],
], { x: 0.6, y: 1.9, w: 12.15, colW: [5.6, 2.6, 1.6, 2.35], rowH: 0.55, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
card(s, 0.6, 4.95, 3.9, 1.9, "Better features help", "0.859 on the small CNN → 0.902 here. Beats random search and, narrowly, the mechanical rule (CIs overlap).", { size: 13 });
card(s, 4.72, 4.95, 3.9, 1.9, "Training wins more clearly", "Gap to the trained probe: 0.045, twice the gap on the small CNN. Hand-picking 26 of 2,048 unnamed features leaves signal unused.", { size: 13 });
card(s, 8.85, 4.95, 3.9, 1.9, "Astra knew", "It predicted ≈ 0.96 and warned that 17 looks at practice data make that optimistic. Practice ran 0.96–0.98; test 0.902.", { size: 13 });
notes(s, "Khaled (1 min)", "We then gave Astra better features: a ResNet-50 trained only on everyday photos, frozen, and asked it to write just the last layer. It used 17 passes and picked 26 of 2,048 features. 0.902, better than on the small CNN, well above random search, and slightly above a mechanical rule that just keeps the top features. But a trained linear probe on the same features gets 0.947. With thousands of unnamed features, training uses all of them; a hand-picked shortlist cannot.");

// ============================================================== P8 robustness (figure)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Robustness check: part of every score is image size, and Astra leans on it more", { placeholder: "title" });
s.addText("A1 found that original image height alone predicts the label (AUROC 0.92). We re-scored everything on 140 test images matched for width and height.", { placeholder: "lead" });
s.addImage({ path: "/home/claude/a2repo/results/figures/project_robust.png", x: 0.6, y: 1.8, w: 7.5, h: 4.9 });
card(s, 8.4, 1.8, 4.35, 1.55, "Everyone drops", "On the matched images, height alone falls to 0.51: the check works. Every model loses AUROC, so the dataset itself rewards this shortcut.", { size: 12.5 });
card(s, 8.4, 3.45, 4.35, 1.75, "Astra drops more", "Small CNN: Astra 0.859 → 0.703 vs trained 0.882 → 0.742.\nResNet-50 features: Astra 0.902 → 0.724 vs trained probe 0.947 → 0.862.", { size: 12.5 });
card(s, 8.4, 5.3, 4.35, 1.55, "Why (a property of the data)", "Pneumonia files are smaller, so after resizing they show less fine texture. A 'less texture' rule reads both disease and file size.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 12 });
notes(s, "Abdulla (1.3 min)", "This is our robustness check. In Assignment 1 we found that the original image size alone predicts pneumonia, because pneumonia files are smaller. So we paired normal and pneumonia test images of the same size, 140 images, where size alone is at chance. Every model drops, so the shortcut is in the data. But Astra's networks drop more than trained ones, on both backbones. The reason is the rule it found: less fine texture. Smaller files look smoother after resizing, so that rule reads disease and file size together. Astra itself warned about this in all three chats.");

// ============================================================== P9 edit mode
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Edit mode (E): fixing our trained ResNet-50 added nothing over a threshold", { placeholder: "title" });
s.addText("Astra edited the last layer (2,048 weights + bias) using practice feedback from one half of the test folder; scored on the other half (312 images).", { placeholder: "lead" });
s.addChart(pres.ChartType.bar, [{ name: "Balanced accuracy", labels: ["Trained ResNet-50, unedited", "Edited by Astra", "Threshold moved on the same practice images"], values: [0.745, 0.814, 0.914] }],
  { x: 0.6, y: 1.85, w: 7.4, h: 4.95, barDir: "bar", barGapWidthPct: 45, chartColors: [X.accent3, X.accent2, X.accent1], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, showLegend: false, showTitle: true, title: "Balanced accuracy (AUROC 0.954 in all three)", ...chartText });
card(s, 8.3, 1.85, 4.45, 2.4, "What Astra did", "Centred the weights, pruned small ones, and lowered the bias by ln 2. AUROC stayed exactly 0.954: the ranking did not change; only the cut-off moved.", { size: 13.5 });
card(s, 8.3, 4.4, 4.45, 2.4, "Lesson", "When a trained model's only flaw is where it draws the line, one number fixes it. MIT's edits removed real obstacles; ours had none.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 13.5 });
notes(s, "Mahd (0.5 min)", "Briefly, the edit mode. Astra edited our trained ResNet-50's last layer to fix false alarms. Balanced accuracy rose from 0.745 to 0.814, but AUROC did not move at all, and simply moving the threshold on the same practice images reached 0.914. When the only flaw is the cut-off, an LLM edit adds nothing.");

// ============================================================== conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
s.addText("CONCLUSION · BOTH PARTS", { placeholder: "kicker" });
s.addText("In both parts, the baseline decided what the number meant", { placeholder: "title" });
s.addText("", { placeholder: "sub" });
card(s, 0.7, 2.9, 5.8, 3.4, "Part 1 · Paper critique", "The ViT's 97.6 % reproduces under a random image split, and a fine-tuned ResNet-50 gets 97.2 % under the same split. With patient grouping and the official test folder, both tie at 0.97 AUROC. The protocol, not the architecture, produced the claim.", { fill: "0B4F5F", headColor: C.accent4, color: "FFFFFF", size: 15 });
card(s, 6.85, 2.9, 5.8, 3.4, "Part 2 · Project", "Yes, with the right feedback: GPT-6 Astra wrote a classifier with no training that beat every matched random search (0.859; 0.902 on ImageNet features). It still trails training (0.882; 0.947), and leans more on an image-size shortcut. Without the random-search and size checks, we would have over-claimed.", { fill: "7A2E1C", headColor: C.accent6, color: "FFFFFF", size: 15 });
notes(s, "Mahd (0.6 min)", "Both parts end in the same place. The ViT's 97.6 meant little without a fair split. Astra's scores meant little without a matched random search and a size check. With them: an LLM can write a working pneumonia classifier without training, and beats random search, but training is still better, and Astra's rule leans more on a dataset shortcut.");

// ============================================================== next
pres.addSection({ title: "Next" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Next" });
s.addText("Limitations, and what we would do next", { placeholder: "title" });
s.addText("Everything here is one dataset and one LLM; the loop was run by hand.", { placeholder: "lead" });
card(s, 0.6, 1.85, 3.9, 3.25, "Limitations", "F: 3 chats; other conditions: 1. Manual loop: one log reset, one accidental re-run (documented; no extra information to Astra). In G, Astra could read the reports with a tool and saw only their top and bottom. Size-matched set: 140 images.", { size: 12.5 });
card(s, 4.72, 1.85, 3.9, 3.25, "Project, next", "Automate the loop; add a second LLM; normalise image size before Astra sees any feedback, to remove the shortcut; report on groups of features in G, so the LLM can use more of them; test on another pediatric hospital.", { size: 12.5 });
card(s, 8.85, 1.85, 3.9, 3.25, "Critique, next", "Close the budget and preprocessing gaps (30 epochs, plain resize, a third seed) and test the ViT where it might win: an external pediatric set and a training-size sweep.", { size: 12.5 });
card(s, 0.6, 5.3, 12.15, 1.55, "Our question for another group", "\"In the paper you critiqued, were the baselines trained with the same budget and the same number of trainable parameters as the proposed model, and how would you check that from the paper alone?\"", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 15 });
notes(s, "Mahd (0.6 min)", "Limitations are on the left: one dataset, one LLM, a manual loop. Next for the project: automate it, try a second LLM, and remove the size shortcut before the LLM sees any feedback. Next for the critique: an external pediatric set and a training-size sweep. Our question for another group is on the slide. Thank you.");
