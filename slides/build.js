const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/ad2037a3-8a2c-4b31-a507-21afa91529d1_dabd3f39-5c18-4f09-8b63-8e8352d5c1e6/pptx/scripts/apply_theme.js");

// Palette: slate ink + white, teal for "ours / fair", coral for "their protocol", grey for "paper reports"
const THEME = {
  name: "CSBP711 Critique", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1C2B36", lt1: "FFFFFF", dk2: "0E3A4A", lt2: "EEF3F5", accent1: "0E7C7B", accent2: "E4572E",
    accent3: "9AA5AE", accent4: "C9E4E3", accent5: "5B6770", accent6: "F3B700", hlink: "0E7C7B", folHlink: "0E7C7B" },
};
const X = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Mahd Hindi, Khaled AlHassani, Abdulla AlKaabi";
pres.title = "CSBP711 Paper Critique and Course Project";
const C = pres.SchemeColor;
const REPO = "github.com/mahdhindi/pneumonia-cxr-algorithm-comparison";
const FOOT = "CSBP711 Paper Critique + Project · Hindi, AlHassani, AlKaabi · code and every result: " + REPO + " (branch a2-paper-critique)";

pres.defineSlideMaster({ title: "DARK", background: { color: X.dk2 }, objects: [
  { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 0.55, w: 12, h: 0.4, fontSize: 12, color: C.accent6, bold: true, charSpacing: 2, margin: 0 } } },
  { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.0, w: 11.9, h: 1.9, fontSize: 40, bold: true, color: C.background1, margin: 0, valign: "top", align: "left" } } },
  { placeholder: { options: { name: "sub", type: "body", x: 0.7, y: 3.0, w: 11.9, h: 0.9, fontSize: 16, color: C.accent4, margin: 0, valign: "top" } } },
  { text: { text: FOOT, options: { x: 0.7, y: 7.0, w: 12, h: 0.3, fontSize: 9, color: C.accent3, margin: 0 } } } ] });
pres.defineSlideMaster({ title: "LIGHT", background: { color: X.lt1 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 28, bold: true, color: C.text2, margin: 0, valign: "top", align: "left" } } },
  { placeholder: { options: { name: "lead", type: "body", x: 0.6, y: 1.2, w: 12.1, h: 0.5, fontSize: 15, color: C.accent5, margin: 0, valign: "top" } } },
  { text: { text: FOOT, options: { x: 0.6, y: 7.05, w: 11.6, h: 0.3, fontSize: 8, color: C.accent3, margin: 0 } } } ],
  slideNumber: { x: 12.5, y: 7.05, w: 0.5, h: 0.3, fontSize: 9, color: C.accent3 } });

// -------------------------------------------------------------- helpers
const box = (s, x, y, w, h, fill, line) => s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: line || fill, width: 0.75 }, rectRadius: 0.08, objectName: "box" });
const txt = (s, t, x, y, w, h, o = {}) => s.addText(t, { x, y, w, h, margin: 0, isTextBox: true, fontSize: 14, color: C.text1, valign: "top", objectName: typeof t === "string" ? t.slice(0, 30) : "runs", ...o });
const card = (s, x, y, w, h, head, body, o = {}) => {
  box(s, x, y, w, h, o.fill || X.lt2, o.line || X.lt2);
  txt(s, head, x + 0.22, y + 0.18, w - 0.44, 0.4, { fontSize: 14, bold: true, color: o.headColor || C.accent1 });
  txt(s, body, x + 0.22, y + 0.6, w - 0.44, h - 0.75, { fontSize: o.size || 13, color: o.color || C.text1, paraSpaceAfter: 4 });
};
const stat = (s, x, y, w, big, label, color) => {
  txt(s, big, x, y, w, 0.85, { fontSize: 40, bold: true, color: color || C.accent1, fontFace: "Cambria" });
  txt(s, label, x, y + 0.9, w, 0.9, { fontSize: 12, color: C.accent5 });
};
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, catAxisLabelColor: X.dk1, valAxisLabelFontSize: 10, valAxisLabelColor: X.accent5, valGridLine: { color: "E3E8EB", size: 0.5 }, catGridLine: { style: "none" },
  dataLabelFontSize: 11, dataLabelColor: X.dk1, legendFontSize: 11, legendColor: X.dk1, titleFontSize: 12, titleColor: X.dk2 };
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: X.dk2 }, fontSize: 11, align: "center", valign: "middle" } });
const td = (t, o = {}) => ({ text: t, options: { fontSize: 11, color: X.dk1, valign: "middle", align: "center", ...o } });
const notes = (s, who, t) => s.addNotes("PRESENTER: " + who + "\n" + t);

// ============================================================== 1 cover
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("CSBP711 ADVANCED ARTIFICIAL INTELLIGENCE · PAPER CRITIQUE + COURSE PROJECT · UAEU · FALL 2026", { placeholder: "kicker" });
s.addText("Two reality checks on\nchildren's pneumonia X-rays", { placeholder: "title" });
s.addText("Part 1 · Paper critique: does a Vision Transformer really beat CNNs? (Singh et al., Scientific Reports 14:2487, 2024)\nPart 2 · Course project: can an LLM write a classifier's weights itself, with no training at all?", { placeholder: "sub" });
txt(s, "Mahd Hindi · Khaled AlHassani · Abdulla AlKaabi", 0.7, 4.4, 10, 0.4, { fontSize: 16, color: C.background1 });
txt(s, "Same dataset and leakage-free pipeline as our Assignment 1 (Kermany pediatric chest X-rays, patient-grouped splits, official test folder held out). About 18 minutes: critique first, then the project.", 0.7, 5.0, 11.5, 1.0, { fontSize: 14, color: C.accent4 });
notes(s, "Mahd (0.3 min)", "Two parts today, on the same dataset and pipeline as our Assignment 1. First, our critique of a paper claiming a Vision Transformer beats CNNs. Then our project: can an LLM write a classifier's weights without any training?");

// ============================================================== 2 the question
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Opening" });
s.addText("The question: two numbers on the same images, 11 points apart", { placeholder: "title" });
s.addText("Assignment 1 gave us a fair, patient-grouped result. The paper reports something much higher. One of two things is true.", { placeholder: "lead" });
box(s, 0.6, 2.0, 5.9, 3.6, X.lt2);
txt(s, "OUR ASSIGNMENT 1", 0.9, 2.2, 5, 0.35, { fontSize: 12, bold: true, color: C.accent1, charSpacing: 1 });
txt(s, "86 %", 0.9, 2.55, 5, 1.1, { fontSize: 60, bold: true, color: C.accent1, fontFace: "Cambria" });
txt(s, "accuracy of the best model (ResNet-50), AUROC 0.971.\nPatient-grouped split, duplicates removed, the dataset's own test folder, two seeds.", 0.9, 3.75, 5.3, 1.6, { fontSize: 14 });
box(s, 6.85, 2.0, 5.9, 3.6, X.lt2);
txt(s, "SINGH ET AL. 2024", 7.15, 2.2, 5, 0.35, { fontSize: 12, bold: true, color: C.accent2, charSpacing: 1 });
txt(s, "97.6 %", 7.15, 2.55, 5, 1.1, { fontSize: 60, bold: true, color: C.accent2, fontFace: "Cambria" });
txt(s, "accuracy of a Vision Transformer (DeiT-Base), AUC 0.966.\nRandom 80/10/10 split, one run, fifteen CNNs reported at 76–94 %.", 7.15, 3.75, 5.3, 1.6, { fontSize: 14 });
card(s, 0.6, 5.85, 12.15, 1.0, "Either the transformer is genuinely better, or the number comes from how it was evaluated.", "Our pipeline can separate the two: run their model under our protocol, and their protocol on our models.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 13 });
notes(s, "Mahd (0.6 min)", "Two numbers on the same images. Our fair protocol gives 86 percent. The paper claims 97.6 with a transformer. Either the transformer is much better than anything we tried, or the evaluation produced the number. We can test which, because we control both the model and the protocol.");

// ============================================================== 3 the paper
pres.addSection({ title: "The paper" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "The paper" });
s.addText("The paper in one line: problem, data, model, claim", { placeholder: "title" });
s.addText("Singh, Kumar, Kumar, Verma, Abhishek & Selvarajan. Scientific Reports 14:2487, January 2024. No code released.", { placeholder: "lead" });
const steps = [["Problem", "Is this child's frontal chest X-ray NORMAL or PNEUMONIA?"], ["Data", "Kermany et al. 2018: 5,856 images, Guangzhou Women and Children's Medical Center, 73 % pneumonia"],
  ["Split", "Random 80 / 10 / 10 by image: 4,684 train, 586 val, 586 test. Patients not mentioned."], ["Model", "DeiT-Base/16 at 224 px, ImageNet weights, fully fine-tuned: Adam, LR 1e-5 × 0.995/epoch, batch 16, 30 epochs"],
  ["Claim", "97.61 % accuracy, sens. 0.949, spec. 0.981, AUC 0.966: \"a great improvement over all other architectures\""]];
steps.forEach(([h, b], i) => {
  const x = 0.6 + i * 2.47;
  s.addShape(pres.ShapeType.ellipse, { x: x + 0.85, y: 1.95, w: 0.7, h: 0.7, fill: { color: i === 4 ? X.accent2 : X.accent1 }, line: { color: i === 4 ? X.accent2 : X.accent1 }, objectName: "step dot" });
  txt(s, String(i + 1), x + 0.85, 2.08, 0.7, 0.5, { fontSize: 18, bold: true, color: "FFFFFF", align: "center" });
  if (i < 4) s.addShape(pres.ShapeType.line, { x: x + 1.6, y: 2.3, w: 0.82, h: 0, line: { color: X.accent3, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
  box(s, x, 2.9, 2.3, 2.6, X.lt2);
  txt(s, h, x + 0.18, 3.05, 2.0, 0.4, { fontSize: 15, bold: true, color: i === 4 ? C.accent2 : C.accent1 });
  txt(s, b, x + 0.18, 3.5, 1.98, 1.95, { fontSize: 12 });
});
card(s, 0.6, 5.75, 12.15, 1.1, "What they offer as evidence", "One training run on one split. Accuracy with a Wald 95 % CI, a confusion matrix, ROC and precision–recall curves, and Table 6: fifteen CNN architectures scoring 76–94 % under what the text calls the same hyperparameters.", { size: 13 });
notes(s, "Khaled (0.8 min)", "The paper in one line. Same dataset as ours. A random 80-10-10 split by image; the word patient never appears. The model is DeiT-Base, 86 million parameters, fully fine-tuned for 30 epochs. The claim: 97.6 percent, a great improvement over all other architectures. The evidence is one run, plus a comparison table of fifteen CNNs.");

// ============================================================== 4 their evidence chart
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "The paper" });
s.addText("Their evidence: Table 6, fifteen CNNs against one transformer", { placeholder: "title" });
s.addText("Accuracy as reported by Singh et al. The ViT is 4.7 points above the best CNN and 15 points above ResNet-50.", { placeholder: "lead" });
const t6 = [["ResNet-101", 75.96], ["ResNet-50", 82.37], ["ResNet-152", 87.18], ["NASNet-L", 88.14], ["DenseNet-169", 88.78], ["ResNet-50V2", 89.26], ["Inception-V3", 89.42], ["VGG-19", 90.22], ["Inc-ResNet-V2", 90.70], ["DenseNet-121", 91.82], ["DenseNet-201", 91.83], ["VGG-16", 92.14], ["ResNet-101V2", 92.62], ["ResNet-152V2", 92.94], ["Quaternion ResNet", 93.75], ["ViT (proposed)", 97.61]];
s.addChart(pres.ChartType.bar, [{ name: "Accuracy % (reported)", labels: t6.map((r) => r[0]), values: t6.map((r) => r[1]) }],
  { x: 0.6, y: 1.8, w: 8.2, h: 5.1, barDir: "bar", barGapWidthPct: 35, chartColors: [X.accent3], invertedColors: [X.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 70, valAxisMaxVal: 100, valAxisMajorUnit: 10, showLegend: false,
    showTitle: true, title: "Accuracy % reported in Table 6 of Singh et al. (2024)", ...chartText, catAxisLabelFontSize: 10 });
card(s, 9.1, 1.8, 3.65, 2.4, "What the table says", "The transformer wins by a wide margin over every CNN, including architectures far larger than it (NASNet-L, 85 M).", { size: 13 });
card(s, 9.1, 4.4, 3.65, 2.5, "What the table also says", "A column most readers skip: trainable parameters. ResNet-50: 200,706. DenseNet-121: 100,354. ViT: 85,800,194. Next slide.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 13 });
notes(s, "Khaled (0.5 min)", "This is their Table 6 as a chart. The transformer at the bottom, 97.6; the best CNN 93.8; ResNet-50 at 82.4. Impressive, until you read one more column of the same table: trainable parameters.");

// ============================================================== 5 weakness 1 frozen
pres.addSection({ title: "Critique" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Critique" });
s.addText("Weak point 1: CNN baselines frozen, transformer fully trained", { placeholder: "title" });
s.addText("Share of each model's parameters that were trained, from the two parameter columns of their Table 6.", { placeholder: "lead" });
const tp = [["ResNet-50", 200706 / (200706 + 23587712) * 100], ["DenseNet-121", 100354 / (100354 + 7037504) * 100], ["ResNet-152V2", 200706 / (200706 + 58331648) * 100], ["NASNet-L", 975746 / (975746 + 84916818) * 100], ["ViT (proposed)", 100]];
s.addChart(pres.ChartType.bar, [{ name: "Trainable share %", labels: tp.map((r) => r[0]), values: tp.map((r) => Math.round(r[1] * 10) / 10) }],
  { x: 0.6, y: 1.8, w: 7.2, h: 4.2, barDir: "bar", barGapWidthPct: 40, chartColors: [X.accent3, X.accent3, X.accent3, X.accent3, X.accent2], invertedColors: [X.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25, showLegend: false,
    showTitle: true, title: "Trainable parameters as % of all parameters (their Table 6)", ...chartText });
stat(s, 8.2, 1.8, 2.2, "0.8 %", "of ResNet-50 was trained: a 200 k head on a frozen 23.6 M backbone", C.accent2);
stat(s, 10.6, 1.8, 2.2, "100 %", "of the ViT was trained: all 85.8 M parameters", C.accent2);
card(s, 8.2, 4.0, 4.55, 2.0, "So the comparison is", "fine-tuning vs feature extraction, not transformer vs CNN. \"All hyperparameters were kept the same\" is true of the optimiser, not of the training.", { size: 13 });
card(s, 0.6, 6.15, 12.15, 0.75, "Also: a 'Ref.' column sits beside every baseline, so some CNN numbers may be cited rather than run. The paper does not say.", "", { size: 12 });
notes(s, "Khaled (0.8 min)", "Weak point one. From their own table: less than one percent of ResNet-50 was trained; a small head on a frozen backbone. The transformer: one hundred percent. That is feature extraction versus fine-tuning. We know from Assignment 1 that a fully fine-tuned ResNet-50 reaches 0.97 AUROC on this data. So the 15-point gap has a simpler explanation than architecture.");

// ============================================================== 6 weakness 2 split
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Critique" });
s.addText("Weak point 2: a random split by image puts one child on both sides", { placeholder: "title" });
s.addText("1,155 children in this dataset have more than one X-ray, up to 30. The paper never mentions patients.", { placeholder: "lead" });
// diagram: random by image
const drawSplit = (x0, title, color, leak) => {
  txt(s, title, x0, 1.85, 5.9, 0.4, { fontSize: 15, bold: true, color });
  box(s, x0, 2.3, 2.75, 2.6, X.lt2); box(s, x0 + 3.15, 2.3, 2.75, 2.6, X.lt2);
  txt(s, "TRAIN", x0 + 0.15, 2.4, 2.4, 0.3, { fontSize: 11, bold: true, color: C.accent5 });
  txt(s, "TEST", x0 + 3.3, 2.4, 2.4, 0.3, { fontSize: 11, bold: true, color: C.accent5 });
  const kids = ["A", "B", "C", "D", "E", "F"];
  // each child has 2-3 scans
  kids.forEach((k, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const cx = x0 + 0.25 + col * 0.85, cy = 2.85 + row * 1.0;
    const inTest = leak ? (i === 1 || i === 4) : (i === 2 || i === 5);
    const n = 2 + (i % 2);
    for (let j = 0; j < n; j++) {
      let sx = cx + j * 0.24, sy = cy;
      let boxX = x0;
      if (leak && inTest && j === n - 1) { boxX = x0 + 3.15; sx = boxX + 0.25 + col * 0.85 + 0.24; }
      else if (!leak && inTest) { boxX = x0 + 3.15; sx = boxX + 0.25 + col * 0.85 + j * 0.24; }
      s.addShape(pres.ShapeType.roundRect, { x: sx, y: sy, w: 0.2, h: 0.28, rectRadius: 0.03, fill: { color: leak && inTest ? X.accent2 : X.accent1 }, line: { color: "FFFFFF", width: 0.5 }, objectName: "scan" });
    }
    if (!(!leak && inTest)) txt(s, "child " + k, cx, cy + 0.32, 0.9, 0.25, { fontSize: 9, color: C.accent5 });
    if (!leak && inTest) txt(s, "child " + k, x0 + 3.15 + 0.25 + col * 0.85, cy + 0.32, 0.9, 0.25, { fontSize: 9, color: C.accent5 });
    if (leak && inTest) txt(s, "child " + k, x0 + 3.15 + 0.25 + col * 0.85 + 0.2, cy + 0.32, 0.9, 0.25, { fontSize: 9, color: C.accent2, bold: true });
  });
};
drawSplit(0.6, "Their split: random by image", C.accent2, true);
drawSplit(6.85, "Our split: whole children", C.accent1, false);
txt(s, "A scan of child B and child E is in the test set while their other scans are in training. The model can recognise the child instead of the disease.", 0.6, 5.0, 5.9, 0.8, { fontSize: 12, color: C.accent5 });
txt(s, "Every scan of a child is on one side only. The model must generalise to children it has never seen.", 6.85, 5.0, 5.9, 0.8, { fontSize: 12, color: C.accent5 });
stat(s, 0.6, 5.55, 3.0, "58 %", "of test images have a sibling scan in training when we reproduce their split (336 of 583)", C.accent2);
stat(s, 3.9, 5.55, 2.8, "0 %", "under our patient-grouped split", C.accent1);
card(s, 6.85, 5.7, 5.9, 1.2, "And their test set comes from the training pool", "The dataset's own test folder is a different collection: in A1 every model scored ≥ 0.99 AUROC on validation but 0.90–0.97 on it. A random 10 % never faces that shift.", { size: 11.5 });
notes(s, "Khaled (1 min)", "Weak point two, the split. Left: random by image. Child B has three scans; two in training, one in test. The model has seen this child. Right: our split keeps every scan of a child on one side. When we reproduced their split, 58 percent of test images had a sibling scan in training. And there is a second issue: their test set is drawn from the training pool, while the dataset's official test folder is a different collection that every model finds harder.");

// ============================================================== 8 verdict before testing
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Critique" });
s.addText("WHAT THE EVIDENCE SUPPORTS, BEFORE WE RUN ANYTHING", { placeholder: "kicker" });
s.addText("The claim is larger than the numbers", { placeholder: "title" });
s.addText("", { placeholder: "sub" });
card(s, 0.7, 2.6, 5.8, 2.6, "Supported", "\"A fully fine-tuned DeiT-Base reaches about 98 % accuracy on a random image-level split of this dataset.\"", { fill: "0B4F5F", headColor: C.accent4, color: "FFFFFF", size: 16 });
card(s, 6.85, 2.6, 5.8, 2.6, "Not supported", "\"A Vision Transformer outperforms CNN architectures on pediatric pneumonia detection.\"", { fill: "7A2E1C", headColor: C.accent6, color: "FFFFFF", size: 16 });
txt(s, "Strengths, for balance: a clear and falsifiable claim, a public dataset, a complete hyper-parameter table, a confusion matrix we can recompute from, and a cost note. That is what makes the claim testable.", 0.7, 5.5, 12, 0.9, { fontSize: 14, color: C.accent4 });
notes(s, "Khaled (0.3 min)", "Our verdict before running anything. Supported: a fine-tuned DeiT reaches about 98 percent on a random image split. Not supported: transformers beat CNNs. To be fair, the paper is clear enough to be tested, which is a strength. Abdulla will show how we tested it.");

// ============================================================== 9 design
pres.addSection({ title: "Reality check" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Reality check" });
s.addText("Reality check design: two models × two protocols, one pipeline", { placeholder: "title" });
s.addText("Everything from Assignment 1 is unchanged. We added their model, and a switch that reproduces their split.", { placeholder: "lead" });
// 2x2 grid
const gx = 0.6, gy = 1.95, cw = 3.1, ch = 1.6;
txt(s, "", gx, gy, 2.2, 0.5);
txt(s, "A1 protocol\npatient-grouped split, official test folder (n = 618)", gx + 2.3, gy, cw, 0.75, { fontSize: 12, bold: true, color: C.accent1, align: "center" });
txt(s, "Paper protocol\nrandom 80/10/10 split by image (n = 583)", gx + 2.3 + cw + 0.15, gy, cw, 0.75, { fontSize: 12, bold: true, color: C.accent2, align: "center" });
const cellG = (r, c, fill, head, body, color) => {
  const x = gx + 2.3 + c * (cw + 0.15), y = gy + 0.85 + r * (ch + 0.15);
  box(s, x, y, cw, ch, fill); txt(s, head, x + 0.2, y + 0.15, cw - 0.4, 0.4, { fontSize: 13, bold: true, color }); txt(s, body, x + 0.2, y + 0.6, cw - 0.4, ch - 0.7, { fontSize: 11.5, color });
};
txt(s, "ViT DeiT-B/16\n(their model)", gx, gy + 0.85 + 0.5, 2.2, 0.8, { fontSize: 13, bold: true, color: C.text2 });
txt(s, "ResNet-50\n(our A1 winner)", gx, gy + 0.85 + ch + 0.15 + 0.5, 2.2, 0.8, { fontSize: 13, bold: true, color: C.text2 });
cellG(0, 0, X.lt2, "NEW · fair test of their model", "Does the ViT beat ResNet-50 when both are fine-tuned under one protocol?", X.dk1);
cellG(0, 1, "FBE9E3", "NEW · reproduce their number", "Do we get their 97.6 % with their model and their split?", X.dk1);
cellG(1, 0, X.lt2, "A1 result", "0.971 AUROC, 86.2 % accuracy. Our reference point.", X.dk1);
cellG(1, 1, "FBE9E3", "NEW · their split on our model", "Does a fine-tuned ResNet-50 also reach ~97 % under their split?", X.dk1);
card(s, 9.6, 1.95, 3.15, 1.55, "Their model, faithfully", "timm deit_base_patch16_224, ImageNet weights, fully fine-tuned, LR 1e-5 as in the paper. 85.8 M params.", { size: 11 });
card(s, 9.6, 3.65, 3.15, 1.55, "Their split, faithfully", "Pool the 5,824 kept images, random label-stratified 80/10/10 by image: 4,659 / 582 / 583. Leakage measured: 58 %.", { size: 11 });
card(s, 9.6, 5.35, 3.15, 1.5, "Held fixed", "Grayscale, pad-to-square, 224 px, same augmentation, AdamW, 12 epochs, best-val-AUROC checkpoint, seeds 42 + 43, one T4.", { size: 11 });
txt(s, "Gaps we could not close: their 30 epochs (we keep 12), plain resize (we pad), Adam (we use AdamW). All three can only help their number; the report lists them.", 0.6, 6.35, 8.8, 0.5, { fontSize: 11, italic: true, color: C.accent5 });
notes(s, "Abdulla (0.8 min)", "Our design is a two-by-two. Rows are the models: their ViT and our ResNet-50. Columns are the protocols: our patient-grouped split with the official test folder, and their random split by image. Three cells are new runs; one is our Assignment 1 result. Their model is DeiT-Base from timm with the paper's own learning rate. Their split we reproduced exactly, and measured its leakage at 58 percent. Everything else is frozen from Assignment 1.");

// ============================================================== 10 result: ViT reproduces
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Reality check" });
s.addText("Result 1: their 97.6 % reproduces, and it depends entirely on the split", { placeholder: "title" });
s.addText("The same ViT, the same pipeline. Only the split changes.", { placeholder: "lead" });
s.addChart(pres.ChartType.bar, [{ name: "ViT accuracy %", labels: ["Reported by the paper\n(their split, 1 run)", "Our run, their split\n(2 seeds)", "Our run, patient-grouped split\n+ official test (2 seeds)"], values: [97.6, 98.0, 82.4] }],
  { x: 0.6, y: 1.8, w: 7.6, h: 5.0, barDir: "col", barGapWidthPct: 55, chartColors: [X.accent3, X.accent2, X.accent1], invertedColors: [X.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 14, valAxisMinVal: 50, valAxisMaxVal: 100, valAxisMajorUnit: 10, showLegend: false,
    showTitle: true, title: "Test accuracy % of DeiT-B/16", ...chartText, dataLabelFontSize: 14 });
stat(s, 8.6, 1.8, 2.0, "98.0 %", "our ViT under their split; they report 97.6 %. Specificity 0.97 vs their 0.98. Reproduced.", C.accent2);
stat(s, 10.8, 1.8, 2.0, "82.4 %", "the same ViT under the fair protocol. Specificity 0.53.", C.accent1);
card(s, 8.6, 4.1, 4.15, 2.75, "Read it as", "The number is real, and we got it. But it needs a test set that shares children with training and is drawn from the same collection. Remove both, and 15.6 points disappear. The model did not change; the evaluation did.", { size: 13 });
notes(s, "Mahd (0.8 min)", "Result one. Grey: the paper. Coral: our run of their model under their split: 98.0, within noise of their 97.6. We reproduced the paper. Teal: the same model under our protocol: 82.4. Fifteen points vanish when children are kept on one side and the official test folder is used. Nothing about the model changed.");

// ============================================================== 11 result: ResNet matches
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Reality check" });
s.addText("Result 2: under their split, a plain ResNet-50 gets the same number", { placeholder: "title" });
s.addText("The 15-point gap in their Table 6 is the frozen backbone, not the architecture.", { placeholder: "lead" });
s.addChart(pres.ChartType.bar, [
  { name: "Reported by the paper", labels: ["ResNet-50", "ViT DeiT-B/16"], values: [82.4, 97.6] },
  { name: "Our run, their split (both fully fine-tuned)", labels: ["ResNet-50", "ViT DeiT-B/16"], values: [97.2, 98.0] },
], { x: 0.6, y: 1.8, w: 7.6, h: 5.0, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [X.accent3, X.accent2],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 50, valAxisMaxVal: 100, valAxisMajorUnit: 10,
  showLegend: true, legendPos: "b", showTitle: true, title: "Test accuracy % under the paper's protocol", ...chartText, dataLabelFontSize: 14 });
stat(s, 8.6, 1.8, 2.0, "97.2 %", "fine-tuned ResNet-50 under their split (two seeds)", C.accent2);
stat(s, 10.8, 1.8, 2.0, "+14.8", "points above the 82.4 % they report for ResNet-50 with a frozen backbone", C.accent2);
card(s, 8.6, 4.1, 4.15, 2.75, "Read it as", "Give the CNN the same treatment as the transformer and the advantage is 0.8 points, inside the seed-to-seed spread. Their comparison measured whether the backbone was trained, and called it architecture.", { size: 13 });
notes(s, "Mahd (0.5 min)", "Result two closes the other gap. Under their own split, a fully fine-tuned ResNet-50 scores 97.2: within a point of the transformer, and fifteen points above the 82.4 they report. That 82.4 is a frozen backbone with a small head. Trained equally, the two architectures are equal.");

// ============================================================== 12 result: fair comparison all models
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Reality check" });
s.addText("Result 3: fair protocol, the ViT ties ResNet-50 at 3.7× the size", { placeholder: "title" });
s.addText("All six models, patient-grouped split, official test folder, mean of seeds 42 and 43. Our Assignment 1 table with the paper's model added.", { placeholder: "lead" });
s.addChart(pres.ChartType.bar, [{ name: "Test AUROC", labels: ["Majority class", "Logistic regression", "Small CNN (scratch)", "DenseNet-121", "ResNet-50", "ViT DeiT-B/16"], values: [0.500, 0.895, 0.942, 0.936, 0.971, 0.973] }],
  { x: 0.6, y: 1.9, w: 6.6, h: 4.9, barDir: "bar", barGapWidthPct: 40, chartColors: [X.accent3, X.accent3, X.accent3, X.accent3, X.accent1, X.accent2], invertedColors: [X.accent3],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.4, valAxisMaxVal: 1.05, valAxisMajorUnit: 0.1, showLegend: false,
    showTitle: true, title: "Test AUROC, fair protocol (higher is better)", ...chartText });
s.addTable([
  [hdr("Model"), hdr("AUROC"), hdr("Acc. %"), hdr("Spec."), hdr("Params"), hdr("Train")],
  [td("ViT DeiT-B/16", { align: "left", bold: true }), td("0.973 ± .001", { bold: true }), td("82.4"), td("0.53"), td("85.8 M"), td("9.2 min")],
  [td("ResNet-50", { align: "left", bold: true }), td("0.971 ± .001", { bold: true }), td("86.2"), td("0.64"), td("23.5 M"), td("4.4 min")],
  [td("Small CNN", { align: "left" }), td("0.942 ± .006"), td("82.6"), td("0.54"), td("0.39 M"), td("3.4 min")],
  [td("DenseNet-121", { align: "left" }), td("0.936 ± .017"), td("82.2"), td("0.53"), td("7.0 M"), td("5.9 min")],
  [td("Logistic regression", { align: "left" }), td("0.895 ± .006"), td("77.4"), td("0.43"), td("0.15 M"), td("3.0 min")],
  [td("Majority class", { align: "left" }), td("0.500"), td("62.6"), td("0.00"), td("0"), td("0")],
], { x: 7.5, y: 1.9, w: 5.25, colW: [1.5, 1.05, 0.65, 0.55, 0.7, 0.8], rowH: 0.34, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
card(s, 7.5, 4.9, 5.25, 2.0, "A tie, at a price", "AUROC 0.973 vs 0.971: the seed-to-seed spread is 0.001, so this is a tie. The transformer uses 3.7× the parameters and 2× the training time, and is worse on accuracy and specificity. Sensitivity is ≥ 0.98 for every model; the differences are all in the healthy children.", { size: 12.5 });
notes(s, "Mahd (0.8 min)", "Result three is the fair comparison, our Assignment 1 ladder with the paper's model added. On AUROC the ViT and ResNet-50 tie at 0.973 and 0.971; the seed spread is 0.001. The transformer costs almost four times the parameters and twice the training time, and is lower on accuracy and specificity. Nothing in this table says 'use a transformer'.");

// ============================================================== 13 why
pres.addSection({ title: "Explanation" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Explanation" });
s.addText("Why: two properties of the data explain the whole gap", { placeholder: "title" });
s.addText("Every model scores ≥ 0.99 AUROC on validation. The drop happens only when the test set is the official folder with unseen children.", { placeholder: "lead" });
s.addChart(pres.ChartType.bar, [
  { name: "Validation AUROC (patient-grouped, same collection as train)", labels: ["Logistic regr.", "Small CNN", "DenseNet-121", "ResNet-50", "ViT DeiT-B"], values: [0.992, 0.998, 1.000, 1.000, 0.999] },
  { name: "Test AUROC (official test folder, unseen children)", labels: ["Logistic regr.", "Small CNN", "DenseNet-121", "ResNet-50", "ViT DeiT-B"], values: [0.895, 0.942, 0.936, 0.971, 0.973] },
], { x: 0.6, y: 1.85, w: 7.2, h: 3.95, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50, chartColors: [X.accent4, X.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 10, valAxisMinVal: 0.8, valAxisMaxVal: 1.02, valAxisMajorUnit: 0.05,
  showLegend: true, legendPos: "b", showTitle: true, title: "Validation vs test AUROC, fair protocol (mean of 2 seeds)", ...chartText, dataLabelFontSize: 10 });
card(s, 8.1, 1.85, 4.65, 2.0, "Property 1 · Patient leakage", "Under their split 58 % of test images have a sibling scan in training. Specificity jumps from 0.53–0.64 to 0.97–0.98 the moment that is allowed. Memorising a child is rewarded as accuracy.", { size: 12 });
card(s, 8.1, 3.95, 4.65, 1.85, "Property 2 · Collection shift", "The official test folder is a different collection (A1 audit: different image geometry and class prior). A random 10 % of the training pool never meets it, so their test is easier by construction.", { size: 12 });
card(s, 0.6, 5.95, 12.15, 1.0, "Plus one property of their setup: frozen baselines.", "Trained equally, ResNet-50 and ViT are within 0.8 points under their protocol and 0.002 AUROC under ours. Architecture did not move the number; the protocol did.", { size: 11.5 });
notes(s, "Abdulla (0.9 min)", "Why. The chart shows every model at 0.99 or better on validation, then dropping on the official test folder. That drop is the collection shift, and it affects the transformer exactly like the CNNs. Property one is leakage: 58 percent of their test images have a sibling scan in training, and specificity jumps to 0.97 the moment that is allowed. Property two is collection shift: a random slice of the training pool never meets the harder folder. Add frozen baselines, and the whole 97.6-versus-82 story is explained with no help from the architecture.");

// ============================================================== 14 verdict
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Explanation" });
s.addText("VERDICT", { placeholder: "kicker" });
s.addText("The claim does not hold up. The protocol, not the architecture, produced the number.", { placeholder: "title" });
s.addText("", { placeholder: "sub" });
card(s, 0.7, 3.1, 3.85, 2.9, "Reproduced", "Their 97.6 % → our 98.0 % under their split. Their evidence is real for the question they asked.", { fill: "0B4F5F", headColor: C.accent4, color: "FFFFFF", size: 14 });
card(s, 4.75, 3.1, 3.85, 2.9, "Explained", "A fine-tuned ResNet-50 gets 97.2 % under the same split. Patient leakage (58 %) and a same-collection test set inflate every model alike.", { fill: "0B4F5F", headColor: C.accent4, color: "FFFFFF", size: 14 });
card(s, 8.8, 3.1, 3.85, 2.9, "What survives", "\"On this dataset the evaluation protocol dominates the architecture.\" Under a fair protocol the ViT ties ResNet-50 at 0.97 AUROC, at 3.7× the size.", { fill: "7A2E1C", headColor: C.accent6, color: "FFFFFF", size: 14 });
notes(s, "Abdulla (0.3 min)", "Verdict. Reproduced, explained, and what survives: on this dataset, protocol dominates architecture. The transformer is fine; the claim about it is not.");

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
s.addText("Result: the channel report brings Astra close to training", { placeholder: "title" });
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
s.addText("Three chats, one rule: less fine texture means pneumonia", { placeholder: "title" });
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
s.addText("Stronger features (G): Astra writes ResNet-50's last layer", { placeholder: "title" });
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
s.addText("Robustness: part of every score is image size", { placeholder: "title" });
s.addText("Astra's rules lean on it more than trained weights do. A1 found that original image height alone predicts the label (AUROC 0.92); we re-scored on 140 size-matched test images.", { placeholder: "lead" });
s.addImage({ path: "/home/claude/a2repo/results/figures/project_robust.png", x: 0.6, y: 1.8, w: 7.5, h: 4.9 });
card(s, 8.4, 1.8, 4.35, 1.55, "Everyone drops", "On the matched images, height alone falls to 0.51: the check works. Every model loses AUROC, so the dataset itself rewards this shortcut.", { size: 12.5 });
card(s, 8.4, 3.45, 4.35, 1.75, "Astra drops more", "Small CNN: Astra 0.859 → 0.703 vs trained 0.882 → 0.742.\nResNet-50 features: Astra 0.902 → 0.724 vs trained probe 0.947 → 0.862.", { size: 12.5 });
card(s, 8.4, 5.3, 4.35, 1.55, "Why (a property of the data)", "Pneumonia files are smaller, so after resizing they show less fine texture. A 'less texture' rule reads both disease and file size.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 12 });
notes(s, "Abdulla (1.3 min)", "This is our robustness check. In Assignment 1 we found that the original image size alone predicts pneumonia, because pneumonia files are smaller. So we paired normal and pneumonia test images of the same size, 140 images, where size alone is at chance. Every model drops, so the shortcut is in the data. But Astra's networks drop more than trained ones, on both backbones. The reason is the rule it found: less fine texture. Smaller files look smoother after resizing, so that rule reads disease and file size together. Astra itself warned about this in all three chats.");

// ============================================================== P9 edit mode (E, H)
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Project" });
s.addText("Edit mode: with the feature report, Astra improved our ResNet-50", { placeholder: "title" });
s.addText("Astra changed the last layer of our fine-tuned ResNet-50. Practice: half the test folder (306 images). All rows scored on the other, harder half (312 children); on all 618 the same model scores 0.966.", { placeholder: "lead" });
s.addTable([
  [hdr("Last layer of our fine-tuned ResNet-50"), hdr("Test AUROC [95 % CI]"), hdr("Balanced acc."), hdr("Size-matched")],
  [td("Unedited", { align: "left" }), td("0.954 [0.925, 0.977]"), td("0.745"), td("0.879")],
  [td("Threshold moved on the practice images", { align: "left" }), td("0.954"), td("0.914"), td("0.879")],
  [td("E · Astra edit, scores only", { align: "left" }), td("0.954"), td("0.814"), td("–")],
  [td("H · Astra edit + per-feature report", { align: "left", bold: true, color: X.accent2 }), td("0.976 [0.959, 0.990]", { bold: true, color: X.accent2 }), td("0.903"), td("0.922")],
  [td("Re-trained on the same 306 images (logistic regression)", { align: "left", bold: true, color: X.accent1 }), td("0.985 [0.974, 0.995]", { bold: true, color: X.accent1 }), td("0.941"), td("0.925")],
], { x: 0.6, y: 1.9, w: 12.15, colW: [5.6, 2.75, 1.9, 1.9], rowH: 0.5, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
card(s, 0.6, 5.1, 3.9, 1.75, "E: scores only", "Astra could only move the cut-off: AUROC stayed 0.954, and a plain threshold move did better.", { size: 13 });
card(s, 4.72, 5.1, 3.9, 1.75, "H: with the report", "It replaced the layer with ±1 votes on 49 features that separate this collection: AUROC 0.954 → 0.976. Retraining still wins (0.985).", { size: 13 });
card(s, 8.85, 5.1, 3.9, 1.75, "Not a size shortcut here", "Size-matched 0.922 vs 0.925 for retraining: fine-tuned features already track the disease (72 images, indicative).", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 13 });
notes(s, "Mahd (0.7 min)", "Finally, edit mode on our own trained ResNet-50. With scores only, condition E, Astra could just move the cut-off and AUROC did not change. With the per-feature report, condition H, it rebuilt the last layer from 49 features that separate this collection, and AUROC rose from 0.954 to 0.976. Retraining the layer on the same 306 images still reaches 0.985. Note that every row is on the same 312 children, a harder half: on all 618 test images the same model scores 0.966, and our A1 two-seed mean was 0.971. And here, unlike the small CNN, Astra's layer does not lean on image size.");

// ============================================================== conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
s.addText("CONCLUSION · BOTH PARTS", { placeholder: "kicker" });
s.addText("In both parts, the baseline decided what the number meant", { placeholder: "title" });
s.addText("", { placeholder: "sub" });
card(s, 0.7, 2.9, 5.8, 3.4, "Part 1 · Paper critique", "The ViT's 97.6 % reproduces under a random image split, and a fine-tuned ResNet-50 gets 97.2 % under the same split. With patient grouping and the official test folder, both tie at 0.97 AUROC. The protocol, not the architecture, produced the claim.", { fill: "0B4F5F", headColor: C.accent4, color: "FFFFFF", size: 15 });
card(s, 6.85, 2.9, 5.8, 3.4, "Part 2 · Project", "Yes, with the right feedback: GPT-6 Astra wrote a classifier with no training that beat every matched random search (0.859; 0.902 on ImageNet features), and lifted our trained ResNet-50 from 0.954 to 0.976. Gradient training with the same information still wins every time (0.882; 0.947; 0.985).", { fill: "7A2E1C", headColor: C.accent6, color: "FFFFFF", size: 15 });
notes(s, "Mahd (0.6 min)", "Both parts end in the same place. The ViT's 97.6 meant little without a fair split. Astra's scores meant little without a matched random search and a size check. With them: an LLM can write a working pneumonia classifier without training, beats random search, and can even improve a trained model, but gradient training with the same information is better every time.");

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

// ============================================================== 16 backup
pres.addSection({ title: "Backup" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Backup" });
s.addText("Backup: every number in one table", { placeholder: "title" });
s.addText("Mean ± std over seeds 42 and 43. n/r = not reported by the paper. Result files: results/*.json in the repository.", { placeholder: "lead" });
s.addTable([
  [hdr("Model"), hdr("Protocol"), hdr("AUROC"), hdr("Accuracy %"), hdr("Sens."), hdr("Spec."), hdr("Params"), hdr("Train")],
  [td("ViT DeiT-B/16 (reported)", { align: "left" }), td("Singh et al., random by image, n=586, 1 run", { align: "left" }), td("0.966"), td("97.6"), td("0.949"), td("0.981"), td("85.8 M"), td("n/r")],
  [td("ViT DeiT-B/16 (ours)", { align: "left" }), td("Singh et al. protocol in our pipeline, n=583", { align: "left" }), td("0.997 ± .002"), td("98.0 ± 0.6"), td("0.984"), td("0.972"), td("85.8 M"), td("9.0 min")],
  [td("ResNet-50 (ours)", { align: "left" }), td("Singh et al. protocol in our pipeline, n=583", { align: "left" }), td("0.995 ± .004"), td("97.2 ± 0.4"), td("0.969"), td("0.978"), td("23.5 M"), td("5.0 min")],
  [td("ResNet-50 (reported)", { align: "left" }), td("Singh et al., backbone frozen", { align: "left" }), td("n/r"), td("82.4"), td("n/r"), td("n/r"), td("0.2 M trainable"), td("n/r")],
  [td("DenseNet-121 (reported)", { align: "left" }), td("Singh et al., backbone frozen", { align: "left" }), td("n/r"), td("91.8"), td("n/r"), td("n/r"), td("0.1 M trainable"), td("n/r")],
  [td("ViT DeiT-B/16 (ours)", { align: "left" }), td("A1: patient-grouped, official test, n=618", { align: "left" }), td("0.973 ± .001"), td("82.4 ± 2.7"), td("0.999"), td("0.530"), td("85.8 M"), td("9.2 min")],
  [td("ResNet-50 (A1)", { align: "left" }), td("A1 protocol", { align: "left" }), td("0.971 ± .001"), td("86.2 ± 3.4"), td("0.997"), td("0.636"), td("23.5 M"), td("4.4 min")],
  [td("DenseNet-121 (A1)", { align: "left" }), td("A1 protocol", { align: "left" }), td("0.936 ± .017"), td("82.2 ± 2.7"), td("0.999"), td("0.526"), td("7.0 M"), td("5.9 min")],
  [td("Small CNN (A1)", { align: "left" }), td("A1 protocol", { align: "left" }), td("0.942 ± .006"), td("82.6 ± 0.3"), td("0.996"), td("0.541"), td("0.39 M"), td("3.4 min")],
  [td("Logistic regression (A1)", { align: "left" }), td("A1 protocol", { align: "left" }), td("0.895 ± .006"), td("77.4 ± 4.2"), td("0.982"), td("0.426"), td("0.15 M"), td("3.0 min")],
  [td("Majority class (A1)", { align: "left" }), td("A1 protocol", { align: "left" }), td("0.500"), td("62.6"), td("1.000"), td("0.000"), td("0"), td("0")],
], { x: 0.6, y: 1.85, w: 12.15, colW: [2.3, 3.6, 1.2, 1.2, 0.85, 0.85, 1.2, 0.95], rowH: 0.38, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
notes(s, "Backup", "Full table for questions.");

// ============================================================== 7 weakness 3 numbers
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Backup" });
s.addText("Backup · weak point 3: numbers that do not reconcile", { placeholder: "title" });
s.addText("Small inconsistencies are not fatal on their own. Together they show the evidence was not checked carefully.", { placeholder: "lead" });
card(s, 0.6, 1.85, 3.9, 2.55, "Table 3 says", "Test set: 330 pneumonia, 256 normal (586 images).", { size: 15 });
txt(s, "≠", 4.6, 2.6, 0.6, 0.8, { fontSize: 40, bold: true, color: C.accent2, align: "center" });
card(s, 5.3, 1.85, 3.9, 2.55, "The confusion matrix says", "TP 152 + FN 8 = 160 pneumonia; TN 420 + FP 6 = 426 normal.\nSame test set, different composition.", { size: 15 });
card(s, 9.4, 1.85, 3.35, 2.55, "Also", "F-score 0.952 (Table 5) vs 0.950 (Table 6) · cross-entropy vs binary cross-entropy · CI 96.2–98.9 % stated, 96.4–98.9 % by their own Eq. 7 · 5,863 images in text, 5,856 in tables", { size: 11.5 });
card(s, 0.6, 4.6, 3.9, 2.3, "One run", "No seed, no cross-validation, no variance. Test accuracy is tracked every epoch next to validation (\"improves along with test accuracy\"), which risks selecting on the test set.", { size: 12.5 });
card(s, 4.7, 4.6, 3.9, 2.3, "Accuracy as headline", "On a set that is 56–73 % pneumonia, a model that always says 'pneumonia' already scores the majority share. AUROC or balanced accuracy would be the honest lead metric.", { size: 12.5 });
card(s, 8.8, 4.6, 3.95, 2.3, "Limitations and reproducibility", "Limitations: one sentence (\"data scarcity and the need for real-world validation\"). No code, no split files, no seeds. The experiment can only be approximated.", { size: 12.5 });
notes(s, "Backup (Khaled, for Q&A)", "Weak point three. Their own tables disagree about what the test set contains. Several smaller inconsistencies. One run, no seed, and the test set watched during training. Accuracy as headline on an imbalanced set. One sentence of limitations and no code. None of this is fatal alone, but it tells you how carefully the evidence was produced. And it leaves a precise, testable gap.");


// ============================================================== backup: project numbers
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Backup" });
s.addText("Backup · project: every number in one table", { placeholder: "title" });
s.addText("Official test folder, n = 618. Astra time = first to last practice pass, including manual copy-paste. Files: results/written_table.md, head_resnet50.md, robust_size.md.", { placeholder: "lead" });
s.addTable([
  [hdr("Weights obtained by"), hdr("Test AUROC"), hdr("Sens."), hdr("Spec."), hdr("Size-matched"), hdr("Params"), hdr("Time")],
  [td("A · Astra, zero-shot", { align: "left" }), td("0.519 [0.47, 0.57]"), td("0.74"), td("0.23"), td("–"), td("5,569"), td("1 reply")],
  [td("C · Astra, 16 images", { align: "left" }), td("0.599 [0.55, 0.65]"), td("0.89"), td("0.14"), td("–"), td("5,569"), td("1 reply")],
  [td("B · Astra, 10 practice passes", { align: "left" }), td("0.737 [0.69, 0.78]"), td("0.59"), td("0.82"), td("0.677"), td("5,569"), td("72 min")],
  [td("D · Astra, images + 11 passes", { align: "left" }), td("0.745 [0.71, 0.79]"), td("0.69"), td("0.72"), td("0.693"), td("5,569"), td("42 min")],
  [td("F · Astra + channel report, runs 1 / 2 / 3", { align: "left" }), td("0.876 / 0.865 / 0.838"), td("0.91"), td("0.50"), td("0.761 / 0.656 / 0.692"), td("5,569"), td("20–30 min")],
  [td("Best of 4 / 10 random nets (median)", { align: "left" }), td("0.691 / 0.716"), td("–"), td("–"), td("–"), td("5,569"), td("seconds")],
  [td("Same CNN, trained (seeds 42, 43)", { align: "left" }), td("0.894 / 0.870"), td("0.97"), td("0.49"), td("0.758 / 0.726"), td("5,569"), td("3.3 min")],
  [td("G · Astra head, frozen ImageNet ResNet-50", { align: "left" }), td("0.902 [0.88, 0.92]"), td("0.94"), td("0.59"), td("0.724"), td("2,049"), td("17 passes")],
  [td("G · trained linear probe, same features", { align: "left" }), td("0.947 [0.93, 0.96]"), td("0.99"), td("0.53"), td("0.862"), td("2,049"), td("< 1 s")],
  [td("ResNet-50 fine-tuned (A1)", { align: "left" }), td("0.971 ± 0.001"), td("1.00"), td("0.64"), td("0.905"), td("23.5 M"), td("4.4 min")],
  [td("H · Astra edit of fine-tuned ResNet-50 (n = 312)", { align: "left" }), td("0.976 [0.96, 0.99]"), td("0.98"), td("0.83"), td("0.922 (72 img)"), td("2,049"), td("7 passes")],
  [td("H · layer re-trained on the 306 practice images", { align: "left" }), td("0.985 [0.97, 1.00]"), td("0.97"), td("0.91"), td("0.925 (72 img)"), td("2,049"), td("< 1 s")],
], { x: 0.6, y: 1.85, w: 12.15, colW: [3.7, 2.05, 0.75, 0.75, 2.0, 1.2, 1.7], rowH: 0.38, border: { type: "solid", color: "E3E8EB", pt: 0.5 }, fill: { color: "FFFFFF" }, fontFace: "Calibri" });
notes(s, "Backup", "All project numbers for questions.");

// ============================================================== backup: honesty log
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Backup" });
s.addText("Backup · how we kept it honest", { placeholder: "title" });
s.addText("Every deviation is logged in astra/README.md; prompts, transcripts, weight files and practice logs are in the repository.", { placeholder: "lead" });
card(s, 0.6, 1.85, 3.9, 2.4, "Could a new chat see earlier runs?", "Model weights do not change between chats. Transcripts log every file Astra opened: each run read only its own prompt and outputs. Each run started from a weak first design (0.59–0.63).", { size: 12.5 });
card(s, 4.72, 1.85, 3.9, 2.4, "Manual-loop slips", "F run 1: Colab reset; passes 3–4 reused the slices of passes 1–2. F run 3: one accidental re-run never shown to Astra. Rivals got the larger budget.", { size: 12.5 });
card(s, 8.85, 1.85, 3.9, 2.4, "Tool use", "A–F: no code run. E: Astra computed weight summaries (mean, RMS) from the weight file only. G: a read-only tool for the long reports (36 Get-Content reads, no computation).", { size: 12.5 });
card(s, 0.6, 4.45, 5.95, 2.4, "Design changes made before Astra saw anything", "E: practice moved from the validation set (model already 0.9995 there) to one half of the test folder; scored on the other half. Practice output never includes image size (A1 showed it leaks the label).", { size: 12.5 });
card(s, 6.8, 4.45, 5.95, 2.4, "What the channel report really is", "Label-informed feature selection: it tells Astra which of its own filters separate the classes. No weight is a formula of the statistics (prompt rule + transcript check), but the method is not data-free, and we say so.", { fill: X.dk2, headColor: C.accent6, color: "FFFFFF", size: 12.5 });
notes(s, "Backup", "For questions about contamination between chats, tool use and deviations.");

(async () => {
  await pres.writeFile({ fileName: "/home/claude/a2deck3/CSBP711_Critique_Project_slides.pptx" });
  await applyTheme("/home/claude/a2deck3/CSBP711_Critique_Project_slides.pptx", THEME);
  console.log("done");
})();
