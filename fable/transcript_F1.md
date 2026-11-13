PRACTICE PASS 1

I have not executed this file or measured the 16 examples numerically (the task says no tools). Every number below comes from looking at the images, so if the file fails to load, send me the error.

What I saw in the examples
Normal lungs are dark with crisp ribs. By eye the gaps between ribs sit around 0.2–0.3 and the ribs around 0.45–0.5, against a mediastinum of 0.65–0.8, with sharp heart and diaphragm borders.
Pneumonia lungs are hazy mid-grey. They sit around 0.4–0.6, the ribs are faint, and the heart and diaphragm borders are blurred. Example 09 is a hard case that looks nearly normal.
Where things sit. Lungs occupy roughly y 45–170, so grid rows 1–2. The right lung is in column 1, the heart and medial left lung in column 2, and columns 0 and 3 are mostly chest wall, arms and background.
A shortcut I am not using. Pneumonia films here tend to be wider, so they get more black padding top and bottom. That is an acquisition artefact and would not transfer to another hospital. I give almost no weight to rows 0 and 3, make the “air” features read zero on pure black, and include one zero-weighted padding channel to measure the confound.
conv1 (5×5, stride 2)
ch	name	detects
0	MEAN	local mean brightness m
1, 2, 3	D50, D35, D20	darkness below 0.50 / 0.35 / 0.20, as relu(1 − m/θ)
4	BLK	near-black (m < 0.06): padding or background
5, 6	B50, B70	brightness above 0.50 / 0.70
7, 8	EHD, EHU	horizontal edges, brighter below / brighter above (rib edges, diaphragm)
9, 10	EVR, EVL	vertical edges, brighter right / brighter left (heart and mediastinal borders)
11–14	diagonals	both diagonal orientations, both polarities (oblique ribs)
15	RIDGE	thin bright line (centre minus ring)

Edge filters are a smoothed derivative with gain 4 and a 0.08 dead zone, so a 0.2 step gives about 0.7.

conv2 (3×3, about a 19 px field)

Two building blocks are used: “air” is capped darkness (D50 − 0.4·D20)/0.6 − 1.5·BLK, which is 1 for lung-dark pixels and at most 0 for pure black. “Dense” is (B50 − 0.6·B70)/0.4, saturating at 1 above 0.7.

ch	combines	final-layer sign
0	smoothed MEAN	+
1–3	smoothed D50, D35, D20	−
4	smoothed BLK (padding monitor)	0
5, 6	smoothed B50, B70	+, 0
7	AERATED: air fraction	−
8	HAZE50: homogeneous mid-grey, 1 − 2.5·(D50 + B50)	+
9	HAZE_SMOOTH: grey 0.35–0.5 with few edges	+
10–13	edge energy: horizontal, vertical, “/”, “”	−
14	RIDGE	−
15, 16	EVR, EVL alone (right / left mediastinal border sharpness)	− at column 1 / column 2
17	EHD alone (diaphragm)	− in rows 2–3
18	RIBS_ON_AIR: air AND edges	−
19	LUNG_INTERIOR: solidly dark 12 px neighbourhood	−
20	air above, dense below (diaphragm silhouette)	− in rows 2–3
21, 22	air-left/dense-right and dense-left/air-right (heart-border silhouettes)	− at columns 1 / 2
23	smooth dark (monitor)	0
24	total edge energy	−
25	bright and smooth (white-out)	+
26	DENSE_GREY: 0.5–0.7 band	+
27	edges on dense tissue (monitor)	0
28–30	coarse brightness gradient: right, left, down	− at heart borders / diaphragm
31	dense fraction	+
Final layer and bias
Cells. Intensity and haze channels are read in rows 1–2 with column weights [0.25, 1, 0.7, 0.25]. Edge channels use [0.15, 1, 0.7, 0.15], because zero padding makes fake edges at the frame border. Diaphragm channels use row 2, and row 3 at 0.6.
Signs. Dark, aerated, crisp-rib and sharp-silhouette evidence votes normal; mid-grey haze, brightness and dense fraction vote pneumonia.
Magnitudes. I chose them so the darkness, haze and edge groups each move the logit by a comparable amount, then scaled everything by 0.4.
Bias. My hand estimate puts a typical normal near −3 and a typical pneumonia near +2, so I set the bias to +0.6 to centre them. This is the least certain number in the file, and I expect to correct it after this pass.
Prediction

I expect an AUROC of about 0.80 on this first pass (plausible range 0.70–0.88). Accuracy at P = 0.5 may be poor until the bias is calibrated.

python
# weights_diag_v1.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    r2 = math.sqrt(2.0)
    EG = 4.0     # edge gain
    ET = 0.08    # edge dead zone

    MEAN, D50, D35, D20, BLK, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, EAP, EAN, EBP, EBN, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D50, 0] = -G / 0.50; b1[D50] = 1.0                # relu(1 - m/0.50)
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[D20, 0] = -G / 0.20; b1[D20] = 1.0                # relu(1 - m/0.20)
    w1[BLK, 0] = -G / 0.06; b1[BLK] = 1.0                # relu(1 - m/0.06)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right
    w1[EAP, 0] = EG * (Dy + Dx) / r2                     # "/" edges, brighter bottom-right
    w1[EAN, 0] = -EG * (Dy + Dx) / r2                    # "/" edges, other polarity
    w1[EBP, 0] = EG * (Dy - Dx) / r2                     # "\" edges, brighter bottom-left
    w1[EBN, 0] = -EG * (Dy - Dx) / r2                    # "\" edges, other polarity
    w1[RDG, 0] = EG * (C3 - RING)                        # thin bright line / blob
    b1[7:16] = -ET

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def capdark(k, coef, pat):
        # (D50 - 0.4*D20)/0.6 : 1 for m <= 0.2, falls to 0 at m = 0.5
        put(k, D50, coef / 0.6, pat)
        put(k, D20, -0.4 * coef / 0.6, pat)

    def air(k, coef, pat):
        # capped darkness that is <= 0 on pure black (padding)
        capdark(k, coef, pat)
        put(k, BLK, -1.5 * coef, pat)

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def edges(k, coef, pat):
        for i in range(7, 15):
            put(k, i, coef, pat)

    # 0-6: smoothed intensity histogram
    put(0, MEAN, 1.0, ALL)
    put(1, D50, 1.0, ALL)
    put(2, D35, 1.0, ALL)
    put(3, D20, 1.0, ALL)
    put(4, BLK, 1.0, ALL)                 # padding monitor (fc weight 0)
    put(5, B50, 1.0, ALL)
    put(6, B70, 1.0, ALL)

    # 7: AERATED - fraction of lung-dark (not black) tissue
    air(7, 1.0, ALL)

    # 8: HAZE50 - homogeneous mid-grey, tent centred on 0.5
    put(8, D50, -2.5, ALL); put(8, B50, -2.5, ALL); b2[8] = 1.0

    # 9: HAZE_SMOOTH - grey 0.35..0.5 with few edges
    put(9, D35, -5.0, ALL); put(9, B50, -3.0, ALL); edges(9, -0.3, ALL); b2[9] = 1.0

    # 10-14: edge energies
    put(10, EHD, 1.0, ALL); put(10, EHU, 1.0, ALL)
    put(11, EVR, 1.0, ALL); put(11, EVL, 1.0, ALL)
    put(12, EAP, 1.0, ALL); put(12, EAN, 1.0, ALL)
    put(13, EBP, 1.0, ALL); put(13, EBN, 1.0, ALL)
    put(14, RDG, 1.0, ALL)

    # 15-17: single-polarity edges for lateralised read-out
    put(15, EVR, 1.0, ALL)
    put(16, EVL, 1.0, ALL)
    put(17, EHD, 1.0, ALL)

    # 18: RIBS_ON_AIR - edges gated by aerated lung (soft AND)
    air(18, 1.0, ALL); edges(18, 0.35, ALL); b2[18] = -1.0

    # 19: LUNG_INTERIOR - neighbourhood solidly dark: relu(2.5*(air - 0.6))
    air(19, 2.5, ALL); b2[19] = -1.5

    # 20: air above / dense below (diaphragm silhouette)
    air(20, 1.0, TOP); dense(20, 1.0, BOT); b2[20] = -1.0

    # 21: air left / dense right (right heart border, left chest wall)
    air(21, 1.0, LEFT); dense(21, 1.0, RIGHT); b2[21] = -1.0

    # 22: dense left / air right (left heart border, right chest wall)
    dense(22, 1.0, LEFT); air(22, 1.0, RIGHT); b2[22] = -1.0

    # 23: SMOOTH_DARK - air without edges (monitor)
    air(23, 1.0, ALL); edges(23, -0.35, ALL)

    # 24: total edge energy
    edges(24, 0.25, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges (white-out)
    dense(25, 1.0, ALL); edges(25, -0.35, ALL)

    # 26: DENSE_GREY - band 0.5..0.7
    put(26, D50, -5.0, ALL); put(26, B70, -2.0, ALL); b2[26] = 1.0

    # 27: EDGES_ON_DENSE - edges where there is no air (monitor)
    edges(27, 0.35, ALL); capdark(27, -1.0, ALL); b2[27] = -0.2

    # 28-30: coarse brightness gradients (offset-invariant contrast)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # 31: DENSE fraction
    dense(31, 1.0, ALL)

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    M = cells([1, 2], [0.25, 1.0, 0.7, 0.25])     # intensity / haze read-out
    E = cells([1, 2], [0.15, 1.0, 0.7, 0.15])     # edge read-out
    L = cells([1, 2], [0.50, 1.0, 0.7, 0.50])     # edge-gated lung read-out
    LOW = cells([2], [0.7, 1.0, 1.0, 0.7]) + cells([3], [0.7, 1.0, 1.0, 0.7], 0.6)

    fc = np.zeros((32, 4, 4))
    fc[0] = 1.0 * M        # brighter lung zone -> pneumonia
    fc[1] = -0.8 * M       # dark lung -> normal
    fc[2] = -0.8 * M
    fc[3] = -0.3 * M
    # fc[4] = 0            # padding monitor, deliberately unused
    fc[5] = 0.4 * M
    # fc[6] = 0            # direction unknown, monitor
    fc[7] = -1.2 * M       # aerated fraction -> normal
    fc[8] = 1.5 * M        # haze -> pneumonia
    fc[9] = 1.0 * M
    fc[10] = -0.5 * E      # crisp ribs -> normal
    fc[11] = -0.2 * E
    fc[12] = -0.25 * E
    fc[13] = -0.25 * E
    fc[14] = -0.3 * E
    fc[15] = -0.5 * cells([1, 2], [0.0, 1.0, 0.0, 0.0])   # sharp right mediastinal border
    fc[16] = -0.4 * cells([1, 2], [0.0, 0.0, 1.0, 0.0])   # sharp left heart border
    fc[17] = -0.3 * LOW                                    # sharp diaphragm
    fc[18] = -0.6 * L      # ribs over aerated lung -> normal
    fc[19] = -0.6 * M
    fc[20] = -1.5 * LOW    # air-over-dense diaphragm silhouette -> normal
    fc[21] = -2.0 * cells([1, 2], [0.0, 1.0, 0.0, 0.3])
    fc[22] = -2.0 * cells([1, 2], [0.3, 0.0, 1.0, 0.0])
    # fc[23] = 0           # monitor
    fc[24] = -0.4 * E
    fc[25] = 0.3 * M
    fc[26] = 0.6 * M
    # fc[27] = 0           # monitor
    fc[28] = -1.0 * cells([1, 2], [0.0, 1.0, 0.0, 0.0])
    fc[29] = -1.0 * cells([1, 2], [0.0, 0.0, 1.0, 0.0])
    fc[30] = -0.5 * LOW
    fc[31] = 0.5 * M

    S = 0.4                                   # global logit scale
    fc_w = (S * fc).reshape(1, 512)
    fc_b = np.array([0.6])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }

# FB-F2. condition F, ONE practice pass. Save Fable's file first in its own cell, starting with
#     %%writefile /content/repo/fable/practice_F1/weights_diag_v1.py     (change run and version)
RUN, N = 1, 1
LLM = "Claude Fable 5.2"
import os; os.chdir('/content/repo'); os.makedirs(f'fable/practice_F{RUN}', exist_ok=True)
f = f'fable/practice_F{RUN}/weights_diag_v{N}.py'
!PYTHONPATH=src python -m cxr.written check --weights {f}
!PYTHONPATH=src python -m cxr.written practice --weights {f} --log fable/practice_log_F{RUN}.jsonl --rotate --per_image --diagnose --budget 20

That is the cell that runs the pass, but its output didn’t come through, so I have nothing to debug from yet. Please paste what the two commands print:

The check result, so I know the file loaded with the right shapes.
The practice result: AUROC, accuracy, sensitivity, specificity, mean P(pneumonia) per class, the per-image table, and the channel diagnostics for the 16 conv1 and 32 conv2 channels.

If the output is too long for one message, the channel diagnostics and the summary metrics matter most; the per-image table can follow separately. Once I have them I’ll send PRACTICE PASS 2.

Pasted text.txt
TXT

PRACTICE PASS 2

What pass 1 showed

Pass 1 scored AUROC 0.773 with specificity 0.17. Most of the logit came from channels that do not separate the classes, while the channels that do separate them had almost no weight.

Edge energy is the real signal. Every edge channel is higher on normals, with separation 0.07–0.08 (about 0.93 as a stand-alone AUROC). Their combined pull on the logit was only about 0.7, though.
Lung-zone brightness carries nothing. conv2.0 contributed +1.000 on normals and +1.013 on pneumonia. Channels 3, 5, 19 and 31 were equally flat, so they only added exposure noise.
My edge-scale guess was about 3× too high. Summed edge energy in lung cells is about 1.0 on normals and 0.6 on pneumonia; I had assumed about 2.8. That left the thresholds in channels 9, 18 and 25 miscalibrated.
The silhouette channels work but were faint. Channels 21 and 22 are about four times stronger on normals, yet contributed under 0.1 logit.
The bias was too high. Nearly every image scored above 0.5.
The padding confound is strong. The near-black channel alone separates at 0.81–0.82, higher on pneumonia. I still give it zero weight.
Changes for pass 2
Zeroed uninformative channels (0, 2, 3, 5, 19, 31), whose lung-cell contributions were the same in both classes.
Made total edge energy (channel 24) the main vote. Its weight goes from −0.16 to −4, read over rows 1–2 with columns [0.3, 1, 1, 0.3]. Column 2 is raised to 1 because the left heart border separated well there.
Raised the other contrast cues whose direction pass 1 confirmed: ridge (14), silhouettes (20, 21, 22), border edges (15–17) and coarse gradients (28–30). Edges on dense tissue (27, separation 0.096) gets a weight for the first time.
Recalibrated edge coefficients inside channels 9, 18 and 25 to the measured edge scale (0.3 → 0.8, 0.35 → 0.8, 0.35 → 0.9).
Kept the weak intensity cues that pointed the right way (1, 7, 8, 9, 25, 26) at about their pass-1 size.
Reset the bias to +8.0. The edge vote is now large and negative, and I placed the decision point at a round lung-cell edge level of about 0.20, between the two levels pass 1 showed. Several terms are on new masks, so this may still be off by a logit or so.
Added probes, at zero or negligible weight, to guide pass 3.
Channels 10–13 now each read one region (top row, bottom row, lateral lung cells, central lung cells) at a weight of −0.05, so their logit columns show where the edge difference lives.
Channel 19 becomes strong-edges-only.
Channel 23 becomes flat non-dark tissue.
Channel 31 becomes edges away from padding.
Prediction

I expect AUROC of about 0.90 on this pass (range 0.86–0.94). Accuracy depends on how close the new bias lands.

python
# weights_diag_v2.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters (unchanged from v1)
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    r2 = math.sqrt(2.0)
    EG = 4.0     # edge gain
    ET = 0.08    # edge dead zone

    MEAN, D50, D35, D20, BLK, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, EAP, EAN, EBP, EBN, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D50, 0] = -G / 0.50; b1[D50] = 1.0                # relu(1 - m/0.50)
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[D20, 0] = -G / 0.20; b1[D20] = 1.0                # relu(1 - m/0.20)
    w1[BLK, 0] = -G / 0.06; b1[BLK] = 1.0                # relu(1 - m/0.06)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right
    w1[EAP, 0] = EG * (Dy + Dx) / r2                     # "/" edges, brighter bottom-right
    w1[EAN, 0] = -EG * (Dy + Dx) / r2                    # "/" edges, other polarity
    w1[EBP, 0] = EG * (Dy - Dx) / r2                     # "\" edges, brighter bottom-left
    w1[EBN, 0] = -EG * (Dy - Dx) / r2                    # "\" edges, other polarity
    w1[RDG, 0] = EG * (C3 - RING)                        # thin bright line / blob
    b1[7:16] = -ET

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def capdark(k, coef, pat):
        # (D50 - 0.4*D20)/0.6 : 1 for m <= 0.2, falls to 0 at m = 0.5
        put(k, D50, coef / 0.6, pat)
        put(k, D20, -0.4 * coef / 0.6, pat)

    def air(k, coef, pat):
        # capped darkness that is <= 0 on pure black (padding)
        capdark(k, coef, pat)
        put(k, BLK, -1.5 * coef, pat)

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def edges(k, coef, pat):
        for i in range(7, 15):
            put(k, i, coef, pat)

    # 0-6: smoothed intensity histogram (0,2,3,5,6 now monitors only)
    put(0, MEAN, 1.0, ALL)
    put(1, D50, 1.0, ALL)
    put(2, D35, 1.0, ALL)
    put(3, D20, 1.0, ALL)
    put(4, BLK, 1.0, ALL)                 # padding monitor (fc weight 0)
    put(5, B50, 1.0, ALL)
    put(6, B70, 1.0, ALL)

    # 7: AERATED - fraction of lung-dark (not black) tissue
    air(7, 1.0, ALL)

    # 8: HAZE50 - homogeneous mid-grey, tent centred on 0.5
    put(8, D50, -2.5, ALL); put(8, B50, -2.5, ALL); b2[8] = 1.0

    # 9: HAZE_SMOOTH - grey 0.35..0.5 with few edges (edge coef recalibrated 0.3 -> 0.8)
    put(9, D35, -5.0, ALL); put(9, B50, -3.0, ALL); edges(9, -0.8, ALL); b2[9] = 1.0

    # 10-14: edge energies
    put(10, EHD, 1.0, ALL); put(10, EHU, 1.0, ALL)
    put(11, EVR, 1.0, ALL); put(11, EVL, 1.0, ALL)
    put(12, EAP, 1.0, ALL); put(12, EAN, 1.0, ALL)
    put(13, EBP, 1.0, ALL); put(13, EBN, 1.0, ALL)
    put(14, RDG, 1.0, ALL)

    # 15-17: single-polarity edges for lateralised read-out
    put(15, EVR, 1.0, ALL)
    put(16, EVL, 1.0, ALL)
    put(17, EHD, 1.0, ALL)

    # 18: RIBS_ON_AIR - edges gated by aerated lung (edge coef recalibrated 0.35 -> 0.8)
    air(18, 1.0, ALL); edges(18, 0.8, ALL); b2[18] = -1.0

    # 19: STRONG_EDGES probe - only above-average edge energy
    edges(19, 0.25, ALL); b2[19] = -0.15

    # 20: air above / dense below (diaphragm silhouette)
    air(20, 1.0, TOP); dense(20, 1.0, BOT); b2[20] = -1.0

    # 21: air left / dense right (right heart border, left chest wall)
    air(21, 1.0, LEFT); dense(21, 1.0, RIGHT); b2[21] = -1.0

    # 22: dense left / air right (left heart border, right chest wall)
    dense(22, 1.0, LEFT); air(22, 1.0, RIGHT); b2[22] = -1.0

    # 23: FLAT_TISSUE probe - no edges and not dark (flat grey / white tissue)
    edges(23, -0.25 / 0.15, ALL); capdark(23, -2.0, ALL); b2[23] = 1.0

    # 24: total edge energy (main read-out)
    edges(24, 0.25, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges (edge coef recalibrated 0.35 -> 0.9)
    dense(25, 1.0, ALL); edges(25, -0.9, ALL)

    # 26: DENSE_GREY - band 0.5..0.7
    put(26, D50, -5.0, ALL); put(26, B70, -2.0, ALL); b2[26] = 1.0

    # 27: EDGES_ON_DENSE - edges where there is no air
    edges(27, 0.35, ALL); capdark(27, -1.0, ALL); b2[27] = -0.2

    # 28-30: coarse brightness gradients (offset-invariant contrast)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # 31: EDGES_NOT_BORDER probe - edge energy away from black padding
    edges(31, 0.25, ALL); put(31, BLK, -2.0, ALL)

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    M = cells([1, 2], [0.25, 1.0, 0.7, 0.25])     # intensity / haze read-out
    E2 = cells([1, 2], [0.30, 1.0, 1.0, 0.30])    # edge / contrast read-out
    L = cells([1, 2], [0.50, 1.0, 0.7, 0.50])     # edge-gated lung read-out
    LOW = cells([2], [0.7, 1.0, 1.0, 0.7]) + cells([3], [0.7, 1.0, 1.0, 0.7], 0.6)
    COL1 = cells([1, 2], [0.0, 1.0, 0.0, 0.0])
    COL2 = cells([1, 2], [0.0, 0.0, 1.0, 0.0])

    fc = np.zeros((32, 4, 4))

    # weak intensity cues that pointed the right way in pass 1
    fc[1] = -0.3 * M       # dark lung -> normal
    fc[7] = -0.5 * M       # aerated fraction -> normal
    fc[8] = 0.6 * M        # haze -> pneumonia
    fc[9] = 1.0 * M        # smooth grey -> pneumonia
    fc[25] = 0.5 * M       # smooth white -> pneumonia
    fc[26] = 0.25 * M      # dense grey -> pneumonia

    # main vote: edge / contrast energy in the lung rows -> normal
    fc[24] = -8.0 * E2
    fc[14] = -5.0 * E2     # thin bright lines (ribs, vessels)
    fc[27] = -2.0 * E2     # edges on dense tissue
    fc[18] = -1.0 * L      # ribs over aerated lung

    # silhouettes and borders -> normal
    fc[20] = -4.0 * LOW                                   # diaphragm silhouette
    fc[21] = -5.0 * cells([1, 2], [0.0, 1.0, 0.0, 0.3])   # right heart border
    fc[22] = -5.0 * cells([1, 2], [0.3, 0.0, 1.0, 0.0])   # left heart border
    fc[15] = -1.0 * COL1
    fc[16] = -1.0 * COL2
    fc[17] = -1.0 * LOW
    fc[28] = -1.0 * E2
    fc[29] = -1.0 * E2
    fc[30] = -1.5 * LOW

    # regional probes (negligible weight): where does the edge difference live?
    fc[10] = -0.1 * cells([0], [1.0, 1.0, 1.0, 1.0])       # top row
    fc[11] = -0.1 * cells([3], [1.0, 1.0, 1.0, 1.0])       # bottom row
    fc[12] = -0.1 * cells([1, 2], [1.0, 0.0, 0.0, 1.0])    # lateral lung cells
    fc[13] = -0.1 * cells([1, 2], [0.0, 1.0, 1.0, 0.0])    # central lung cells

    # zero weight: 0, 2, 3, 5, 6 (no class difference), 4 (padding), 19, 23, 31 (probes)

    S = 0.5                                   # global logit scale
    fc_w = (S * fc).reshape(1, 512)
    fc_b = np.array([8.0])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 3

What pass 2 showed

Pass 2 reached AUROC 0.899 (accuracy 0.83, sensitivity 0.83, specificity 0.84), so the edge-dominant readout and the new bias both worked. Three findings shape pass 3:

My lung-zone readout still trails the whole-image edge statistic. On the same 256 images, channel 24’s whole-image mean separates at 0.075 (about 0.925 as an AUROC), better than my full combination at 0.899. Either rows 0 and 3 carry real signal, or the extra terms I stacked on top are diluting it.
The regional probes disagree by region. Pneumonia edge energy as a fraction of normal is 0.60 in the central lung cells, 0.79 in the lateral lung cells, 0.63 in the top row and 0.43 in the bottom row. The bottom-row figure is suspicious, because that row is where the padding sits.
Other edge variants are no better. Strong-edges-only (0.105) is worse than plain edge energy. Removing padding-border edges (0.070) is slightly better. Flat-tissue (0.651) is weak.

The mis-ranked cases are about 20 pneumonia films scored below 0.2 (high-contrast films) and a handful of blurry normals. No intensity cue I have is strong enough to rescue them yet.

What pass 3 does

This pass is a controlled test, not an attempt at a higher score. Everything is stripped to one term so I can attribute the gap.

Pure readout. The logit is total edge energy (channel 24) in rows 1–2 with columns [0.3, 1, 1, 0.3], weight −8, and nothing else. Comparing its AUROC with channel 24’s whole-image separation on the same images tells me two things: whether the lung rows alone match the whole image, and whether pass 2’s extras helped or hurt.
Bias +8.8. This keeps the decision point at the same lung-cell edge level as pass 2 (about 0.21), nudged slightly toward pneumonia.
Regional probes. Nine spare channels (0, 2, 3, 5, 6, 10–13) become copies of total edge energy, each read in one region at a weight of −0.1. Their logit columns will show the class means for each central lung cell, each lateral column, the top-centre, the bottom-centre and the corners. That will set the cell mask.
Padding-imputation probes. Channels 19 and 23 are edge energy plus 0.25 or 0.35 times the near-black channel, which counts padding as average tissue. If their whole-image separation stays near 0.07, the whole-image edge signal is real edge density and I can use all rows safely. If it degrades, the whole-image advantage is partly the padding shortcut and I stay with the lung rows.
Padding-intrusion probe. The near-black channel gets a weight of −0.05 in rows 1–2, signed against the shortcut, to show how much black reaches the lung rows in each class.

All other channels keep their definitions but have zero weight this pass. Their separations are still reported.

Prediction

I expect AUROC of about 0.90 (range 0.87–0.93), probably just below channel 24’s whole-image figure.

python
# weights_diag_v3.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters (unchanged)
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    r2 = math.sqrt(2.0)
    EG = 4.0     # edge gain
    ET = 0.08    # edge dead zone

    MEAN, D50, D35, D20, BLK, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, EAP, EAN, EBP, EBN, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D50, 0] = -G / 0.50; b1[D50] = 1.0                # relu(1 - m/0.50)
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[D20, 0] = -G / 0.20; b1[D20] = 1.0                # relu(1 - m/0.20)
    w1[BLK, 0] = -G / 0.06; b1[BLK] = 1.0                # relu(1 - m/0.06)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right
    w1[EAP, 0] = EG * (Dy + Dx) / r2                     # "/" edges, brighter bottom-right
    w1[EAN, 0] = -EG * (Dy + Dx) / r2                    # "/" edges, other polarity
    w1[EBP, 0] = EG * (Dy - Dx) / r2                     # "\" edges, brighter bottom-left
    w1[EBN, 0] = -EG * (Dy - Dx) / r2                    # "\" edges, other polarity
    w1[RDG, 0] = EG * (C3 - RING)                        # thin bright line / blob
    b1[7:16] = -ET

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def capdark(k, coef, pat):
        # (D50 - 0.4*D20)/0.6 : 1 for m <= 0.2, falls to 0 at m = 0.5
        put(k, D50, coef / 0.6, pat)
        put(k, D20, -0.4 * coef / 0.6, pat)

    def air(k, coef, pat):
        # capped darkness that is <= 0 on pure black (padding)
        capdark(k, coef, pat)
        put(k, BLK, -1.5 * coef, pat)

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def edges(k, coef, pat):
        for i in range(7, 15):
            put(k, i, coef, pat)

    # Regional probe copies of total edge energy (same formula as channel 24)
    for k in (0, 2, 3, 5, 6, 10, 11, 12, 13):
        edges(k, 0.25, ALL)

    # 1: smoothed D50
    put(1, D50, 1.0, ALL)

    # 4: smoothed near-black (padding)
    put(4, BLK, 1.0, ALL)

    # 7: AERATED - fraction of lung-dark (not black) tissue
    air(7, 1.0, ALL)

    # 8: HAZE50 - homogeneous mid-grey, tent centred on 0.5
    put(8, D50, -2.5, ALL); put(8, B50, -2.5, ALL); b2[8] = 1.0

    # 9: HAZE_SMOOTH - grey 0.35..0.5 with few edges
    put(9, D35, -5.0, ALL); put(9, B50, -3.0, ALL); edges(9, -0.8, ALL); b2[9] = 1.0

    # 14: ridge energy
    put(14, RDG, 1.0, ALL)

    # 15-17: single-polarity edges
    put(15, EVR, 1.0, ALL)
    put(16, EVL, 1.0, ALL)
    put(17, EHD, 1.0, ALL)

    # 18: RIBS_ON_AIR - edges gated by aerated lung
    air(18, 1.0, ALL); edges(18, 0.8, ALL); b2[18] = -1.0

    # 19: imputation probe - edge energy with padding counted as 0.25
    edges(19, 0.25, ALL); put(19, BLK, 0.25, ALL)

    # 20: air above / dense below (diaphragm silhouette)
    air(20, 1.0, TOP); dense(20, 1.0, BOT); b2[20] = -1.0

    # 21: air left / dense right (right heart border, left chest wall)
    air(21, 1.0, LEFT); dense(21, 1.0, RIGHT); b2[21] = -1.0

    # 22: dense left / air right (left heart border, right chest wall)
    dense(22, 1.0, LEFT); air(22, 1.0, RIGHT); b2[22] = -1.0

    # 23: imputation probe - edge energy with padding counted as 0.35
    edges(23, 0.25, ALL); put(23, BLK, 0.35, ALL)

    # 24: total edge energy (the only real read-out this pass)
    edges(24, 0.25, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges
    dense(25, 1.0, ALL); edges(25, -0.9, ALL)

    # 26: DENSE_GREY - band 0.5..0.7
    put(26, D50, -5.0, ALL); put(26, B70, -2.0, ALL); b2[26] = 1.0

    # 27: EDGES_ON_DENSE - edges where there is no air
    edges(27, 0.35, ALL); capdark(27, -1.0, ALL); b2[27] = -0.2

    # 28-30: coarse brightness gradients
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # 31: EDGES_NOT_BORDER - edge energy away from black padding
    edges(31, 0.25, ALL); put(31, BLK, -2.0, ALL)

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E2 = cells([1, 2], [0.30, 1.0, 1.0, 0.30])    # lung-row edge read-out

    fc = np.zeros((32, 4, 4))

    # the single real term: edge energy in the lung rows -> normal
    fc[24] = -8.0 * E2

    # regional probes of the same quantity (negligible weight)
    P = -0.1
    fc[0] = P * cells([1], [0.0, 1.0, 0.0, 0.0])           # upper right lung (row 1, col 1)
    fc[2] = P * cells([1], [0.0, 0.0, 1.0, 0.0])           # upper heart / left lung (row 1, col 2)
    fc[3] = P * cells([2], [0.0, 1.0, 0.0, 0.0])           # lower right lung (row 2, col 1)
    fc[5] = P * cells([2], [0.0, 0.0, 1.0, 0.0])           # lower heart / left lung (row 2, col 2)
    fc[6] = P * cells([1, 2], [1.0, 0.0, 0.0, 0.0])        # lateral column 0, lung rows
    fc[10] = P * cells([1, 2], [0.0, 0.0, 0.0, 1.0])       # lateral column 3, lung rows
    fc[11] = P * cells([0], [0.0, 1.0, 1.0, 0.0])          # top row, centre
    fc[12] = P * cells([3], [0.0, 1.0, 1.0, 0.0])          # bottom row, centre
    fc[13] = P * cells([0, 3], [1.0, 0.0, 0.0, 1.0])       # four corners

    # padding-intrusion probe: how much black reaches the lung rows
    fc[4] = -0.05 * cells([1, 2], [1.0, 1.0, 1.0, 1.0])

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([8.8])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 4

What pass 3 showed

The pure lung-row edge readout scored AUROC 0.881 (accuracy 0.82, sensitivity 0.79, specificity 0.88). The probes settled most of what I asked:

Pass 2’s extra terms did nothing. The pure readout trails the whole-image edge statistic by 0.023 on this batch (0.881 against 0.904). Pass 2, with all its extras, trailed by 0.026.
The whole-image advantage looks tied to padding. Counting padding as average tissue (channels 19 and 23) drops whole-image separation from 0.904 to 0.80 and 0.73. That probe still includes padding-border edges, so it is suggestive, not conclusive. I am staying with the lung rows.
The lung rows are free of padding. Near-black averages about 0.10 there on normals and 0.08 on pneumonia, so what black there is comes from background beside the patient.
The central lung cells carry the signal. Pneumonia edge energy is 0.50–0.64 of normal in the four central cells, but 0.74–0.81 in the lateral columns. The top-centre shows the biggest gap, but that is where the padding sits.

This batch was harder overall: every edge channel separated worse than in passes 1 and 2.

Changes for pass 4
Lateral columns down-weighted. The edge mask becomes rows 1–2 with columns [0.15, 1, 1, 0.15], because the lateral cells separate poorly and hold background and arms.
Coarse lung-to-mediastinum contrast added at meaningful weight. Channels 28 and 29 (brightness gradient across about 12 px, rightward and leftward) get −2.5 on the lung mask, and channel 30 (downward, the diaphragm) gets −2. Their whole-image separation matches the fine edges (0.09–0.11), and they measure a different thing: how dark the lung is against the heart, not how crisp the ribs are. In pass 2 they were too small to matter.
Ridge channel 14 restored at −6 as a third contrast scale.
Fine-edge weight set to −6 and bias to +12.0, to keep the decision point centred between the class levels seen so far, slightly toward pneumonia.
Edge-detector experiment in conv1. Three intensity filters that are unused this pass (2, 5 and 6) become variants of the best edge filter, channel 10: one with no dead zone, one with a tripled dead zone, and one with a finer 3-tap kernel. Their separations sit next to channel 10’s on the same images, so I can see whether the edge filter itself can be improved.
Regional probes for the coarse channels. Nine spare conv2 channels read the rightward, leftward and downward gradients in separate regions at a weight of −0.1, to set their masks next pass. Channel 8 becomes an upward-gradient probe.

The conv2 channels that depended on the three borrowed filters (9, 20–22, 25, 26) are zeroed for this pass only. Channel 24 remains as the on-batch reference.

Prediction

I expect AUROC of about 0.91 (range 0.88–0.93). This batch is mostly pass-1 images, where the whole-image edge reference was 0.934.

python
# weights_diag_v4.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    d3 = np.array([0.0, -1.0, 0.0, 1.0, 0.0])            # fine 3-tap derivative, unit step -> 1
    Dx_fine = np.outer(h, d3)                            # + when brighter to the RIGHT (fine)
    r2 = math.sqrt(2.0)
    EG = 4.0     # edge gain
    ET = 0.08    # edge dead zone

    MEAN, D50, PV1, D20, BLK, PV2, PV3 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, EAP, EAN, EBP, EBN, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D50, 0] = -G / 0.50; b1[D50] = 1.0                # relu(1 - m/0.50)
    w1[D20, 0] = -G / 0.20; b1[D20] = 1.0                # relu(1 - m/0.20)
    w1[BLK, 0] = -G / 0.06; b1[BLK] = 1.0                # relu(1 - m/0.06)

    # edge-detector variants of channel 10 (bright left, dark right), probes only
    w1[PV1, 0] = -EG * Dx;      b1[PV1] = 0.0            # no dead zone
    w1[PV2, 0] = -EG * Dx;      b1[PV2] = -3.0 * ET      # tripled dead zone
    w1[PV3, 0] = -EG * Dx_fine; b1[PV3] = -ET            # finer 3-tap kernel

    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right
    w1[EAP, 0] = EG * (Dy + Dx) / r2                     # "/" edges, brighter bottom-right
    w1[EAN, 0] = -EG * (Dy + Dx) / r2                    # "/" edges, other polarity
    w1[EBP, 0] = EG * (Dy - Dx) / r2                     # "\" edges, brighter bottom-left
    w1[EBN, 0] = -EG * (Dy - Dx) / r2                    # "\" edges, other polarity
    w1[RDG, 0] = EG * (C3 - RING)                        # thin bright line / blob
    b1[7:16] = -ET

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def capdark(k, coef, pat):
        # (D50 - 0.4*D20)/0.6 : 1 for m <= 0.2, falls to 0 at m = 0.5
        put(k, D50, coef / 0.6, pat)
        put(k, D20, -0.4 * coef / 0.6, pat)

    def air(k, coef, pat):
        # capped darkness that is <= 0 on pure black (padding)
        capdark(k, coef, pat)
        put(k, BLK, -1.5 * coef, pat)

    def edges(k, coef, pat):
        for i in range(7, 15):
            put(k, i, coef, pat)

    def coarse_r(k):   # brighter to the right across ~12 px
        put(k, MEAN, 3.0, RIGHT); put(k, MEAN, -3.0, LEFT)

    def coarse_l(k):   # brighter to the left
        put(k, MEAN, 3.0, LEFT); put(k, MEAN, -3.0, RIGHT)

    def coarse_d(k):   # brighter below
        put(k, MEAN, 3.0, BOT); put(k, MEAN, -3.0, TOP)

    def coarse_u(k):   # brighter above
        put(k, MEAN, 3.0, TOP); put(k, MEAN, -3.0, BOT)

    # regional probe copies of the coarse gradients
    coarse_r(0); coarse_r(2); coarse_r(6)
    coarse_l(3); coarse_l(5); coarse_l(10)
    coarse_d(11); coarse_d(12); coarse_d(13)

    # 1: smoothed D50
    put(1, D50, 1.0, ALL)

    # 4: smoothed near-black (padding monitor)
    put(4, BLK, 1.0, ALL)

    # 7: AERATED - fraction of lung-dark (not black) tissue
    air(7, 1.0, ALL)

    # 8: upward coarse gradient probe
    coarse_u(8)

    # 9: zero this pass (its conv1 input is borrowed)

    # 14: ridge energy
    put(14, RDG, 1.0, ALL)

    # 15-17: single-polarity edges
    put(15, EVR, 1.0, ALL)
    put(16, EVL, 1.0, ALL)
    put(17, EHD, 1.0, ALL)

    # 18: RIBS_ON_AIR - edges gated by aerated lung
    air(18, 1.0, ALL); edges(18, 0.8, ALL); b2[18] = -1.0

    # 19: strong edges only (reference)
    edges(19, 0.25, ALL); b2[19] = -0.15

    # 20, 21, 22: zero this pass (their conv1 inputs are borrowed)

    # 23: smoothed pass-through of the no-dead-zone edge variant
    put(23, PV1, 1.0, ALL)

    # 24: total edge energy (main read-out and on-batch reference)
    edges(24, 0.25, ALL)

    # 25, 26: zero this pass (their conv1 inputs are borrowed)

    # 27: EDGES_ON_DENSE - edges where there is no air
    edges(27, 0.35, ALL); capdark(27, -1.0, ALL); b2[27] = -0.2

    # 28-30: coarse brightness gradients (lung-to-mediastinum contrast)
    coarse_r(28)
    coarse_l(29)
    coarse_d(30)

    # 31: EDGES_NOT_BORDER - edge energy away from black padding (reference)
    edges(31, 0.25, ALL); put(31, BLK, -2.0, ALL)

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral
    LOW = cells([2], [0.7, 1.0, 1.0, 0.7]) + cells([3], [0.7, 1.0, 1.0, 0.7], 0.6)

    fc = np.zeros((32, 4, 4))

    # fine contrast: rib and vessel edges in the lung rows -> normal
    fc[24] = -6.0 * E4
    fc[14] = -6.0 * E4

    # coarse contrast: dark lung against heart, mediastinum, diaphragm -> normal
    fc[28] = -2.5 * E4
    fc[29] = -2.5 * E4
    fc[30] = -2.0 * LOW

    # regional probes of the coarse gradients (negligible weight)
    P = -0.1
    fc[0] = P * cells([1, 2], [0.0, 1.0, 0.0, 0.0])        # rightward, column 1
    fc[2] = P * cells([1, 2], [0.0, 0.0, 1.0, 0.0])        # rightward, column 2
    fc[6] = P * cells([1, 2], [1.0, 0.0, 0.0, 1.0])        # rightward, lateral columns
    fc[3] = P * cells([1, 2], [0.0, 1.0, 0.0, 0.0])        # leftward, column 1
    fc[5] = P * cells([1, 2], [0.0, 0.0, 1.0, 0.0])        # leftward, column 2
    fc[10] = P * cells([1, 2], [1.0, 0.0, 0.0, 1.0])       # leftward, lateral columns
    fc[11] = P * cells([1], [0.0, 1.0, 1.0, 0.0])          # downward, row 1 centre
    fc[12] = P * cells([2], [0.0, 1.0, 1.0, 0.0])          # downward, row 2 centre
    fc[13] = P * cells([3], [0.0, 1.0, 1.0, 0.0])          # downward, row 3 centre

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([12.0])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 5

What pass 4 showed

Pass 4 scored AUROC 0.918 (accuracy 0.85, sensitivity 0.84, specificity 0.86), the best so far. Its gap to the whole-image edge reference on the same images narrowed to 0.013, from 0.023–0.026 in passes 2 and 3. So the refined mask and the added coarse contrast together helped.

The edge dead zone is hurting. The no-dead-zone variant separates at 0.068 against 0.076 for the standard filter. Tripling the dead zone makes it worse (0.102), and the finer 3-tap kernel changes nothing (0.078). Weak gradients carry signal.
Coarse contrast is informative only at the mediastinal borders.
In the central columns pneumonia sits at 0.56–0.74 of normal.
In the lateral columns it sits at 0.84–0.87, with large values in both classes, so those cells mostly add noise.
The downward gradient is useful in row 2 only.
Row 2 (the diaphragm) sits at 0.67 of normal, while row 1 is weak (0.89).
Row 3 shows the biggest gap (0.29), but it is the padded row, so I am removing it as a possible shortcut.
Changes for pass 5
Edge dead zone removed on all eight edge filters (bias −0.08 to 0). The ridge filter keeps its dead zone because I have not tested it without.
Coarse masks tightened. The rightward and leftward gradients are read only in the four central lung cells, at −3. The downward gradient is read only in row 2, with columns [0.5, 1, 1, 0.5], at −3.
Fine-edge weight set to −5 and bias to +13.3. Removing the dead zone raises all edge values, so the centre of the logit moves; this estimate could be off by half a logit.
A cleaner test of the padding question. Vertical edges cannot see the top and bottom padding borders, which confounded pass 3’s test. Channels 20–22 are vertical-edge energy plus 0.28, 0.20 or 0.36 times near-black, next to the plain version in channel 11. If counting padding as tissue leaves separation intact, the whole-image signal is real edge density and I can safely use more of the frame. If it collapses again, the whole-image advantage is the shortcut.
Capped-edge probes. Strong edges separate worse than weak ones, so channels 23 and 19 compute edge energy capped at two levels, to see whether limiting strong edges (bones, wires, borders) beats the plain sum in channel 16.
Fine-detail probe. conv1 filter 6 is a pixel-level Laplacian, to check whether the signal is really image grain and not anatomy.

The haze and silhouette channels stay switched off. They did not measurably help in pass 2, and their conv1 inputs are in use as probes.

Prediction

I expect AUROC of about 0.92 (range 0.89–0.94). This batch is mostly pass-2 images, where the earlier model scored 0.899 and the whole-image reference was 0.925.

python
# weights_diag_v5.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    LAP = np.zeros((5, 5))                               # pixel-level Laplacian
    LAP[2, 2] = 1.0
    LAP[1, 2] = -0.25; LAP[3, 2] = -0.25; LAP[2, 1] = -0.25; LAP[2, 3] = -0.25
    r2 = math.sqrt(2.0)
    EG = 4.0     # edge gain

    MEAN, D50, PVA, D20, BLK, PVB, PVL = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, EAP, EAN, EBP, EBN, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D50, 0] = -G / 0.50; b1[D50] = 1.0                # relu(1 - m/0.50)
    w1[D20, 0] = -G / 0.20; b1[D20] = 1.0                # relu(1 - m/0.20)
    w1[BLK, 0] = -G / 0.06; b1[BLK] = 1.0                # relu(1 - m/0.06)

    # probes: thresholded copies of channel 10 (used to build capped edges), and fine detail
    w1[PVA, 0] = -EG * Dx; b1[PVA] = -0.24               # channel 10 above 0.24
    w1[PVB, 0] = -EG * Dx; b1[PVB] = -0.50               # channel 10 above 0.50
    w1[PVL, 0] = EG * LAP; b1[PVL] = 0.0                 # pixel-level bright detail

    # edge filters, dead zone removed
    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right
    w1[EAP, 0] = EG * (Dy + Dx) / r2                     # "/" edges, brighter bottom-right
    w1[EAN, 0] = -EG * (Dy + Dx) / r2                    # "/" edges, other polarity
    w1[EBP, 0] = EG * (Dy - Dx) / r2                     # "\" edges, brighter bottom-left
    w1[EBN, 0] = -EG * (Dy - Dx) / r2                    # "\" edges, other polarity
    w1[RDG, 0] = EG * (C3 - RING); b1[RDG] = -0.08       # thin bright line / blob

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def capdark(k, coef, pat):
        # (D50 - 0.4*D20)/0.6 : 1 for m <= 0.2, falls to 0 at m = 0.5
        put(k, D50, coef / 0.6, pat)
        put(k, D20, -0.4 * coef / 0.6, pat)

    def air(k, coef, pat):
        # capped darkness that is <= 0 on pure black (padding)
        capdark(k, coef, pat)
        put(k, BLK, -1.5 * coef, pat)

    def edges(k, coef, pat):
        for i in range(7, 15):
            put(k, i, coef, pat)

    def vert(k, coef, pat):
        put(k, EVR, coef, pat); put(k, EVL, coef, pat)

    # 1, 4, 7: intensity monitors (zero weight)
    put(1, D50, 1.0, ALL)
    put(4, BLK, 1.0, ALL)                 # padding monitor
    air(7, 1.0, ALL)

    # 8: upward coarse gradient (monitor)
    put(8, MEAN, 3.0, TOP); put(8, MEAN, -3.0, BOT)

    # 9: smoothed fine-detail probe
    put(9, PVL, 1.0, ALL)

    # 11: vertical-edge energy (reference for the imputation probes)
    vert(11, 1.0, ALL)

    # 14: ridge energy
    put(14, RDG, 1.0, ALL)

    # 15-17: single-polarity edges (16 is the reference for the capped probes)
    put(15, EVR, 1.0, ALL)
    put(16, EVL, 1.0, ALL)
    put(17, EHD, 1.0, ALL)

    # 19, 23: capped-edge probes, min(channel-10 response, cap)
    put(19, EVL, 1.0, ALL); put(19, PVB, -1.0, ALL)      # capped at 0.50
    put(23, EVL, 1.0, ALL); put(23, PVA, -1.0, ALL)      # capped at 0.24

    # 20-22: imputation probes, vertical edges with black counted as tissue
    vert(20, 1.0, ALL); put(20, BLK, 0.28, ALL)
    vert(21, 1.0, ALL); put(21, BLK, 0.20, ALL)
    vert(22, 1.0, ALL); put(22, BLK, 0.36, ALL)

    # 24: total edge energy (main read-out and on-batch reference)
    edges(24, 0.25, ALL)

    # 28-30: coarse brightness gradients (lung-to-mediastinum contrast)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # 31: edge energy away from black padding (reference)
    edges(31, 0.25, ALL); put(31, BLK, -2.0, ALL)

    # channels 0, 2, 3, 5, 6, 10, 12, 13, 18, 25, 26, 27 are unused (all zero) this pass

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral
    C4 = cells([1, 2], [0.0, 1.0, 1.0, 0.0])      # central lung cells only
    DIA = cells([2], [0.5, 1.0, 1.0, 0.5])        # diaphragm row

    fc = np.zeros((32, 4, 4))

    # fine contrast: rib and vessel edges in the lung rows -> normal
    fc[24] = -5.0 * E4
    fc[14] = -6.0 * E4

    # coarse contrast: dark lung against heart, mediastinum, diaphragm -> normal
    fc[28] = -3.0 * C4
    fc[29] = -3.0 * C4
    fc[30] = -3.0 * DIA

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([13.3])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 6

What pass 5 showed

Pass 5 scored AUROC 0.902 (accuracy 0.85, sensitivity 0.85, specificity 0.84). That is no better than pass 2’s 0.899 on largely the same images, so I have to correct my reading of pass 4.

Pass 4’s gain was mostly the batch. On pass-2 images the refined model gains only 0.003 over the pass-2 model. The lung-row contrast readout sits at about 0.90 on this batch whatever I do to masks and scales.
The whole-image advantage is the padding shortcut. With the border confound removed, counting black as tissue takes vertical-edge separation from 0.929 to 0.915, 0.892 and 0.854 as the imputed level rises. At a fair level the whole image is worse than the lung rows, so I am keeping the lung rows.
Capping strong edges helps a little. Capped versions separate at 0.060–0.063 against 0.071 for the plain filter. I am not adopting this yet because it needs a conv1 rebuild.
The signal is not image grain. The pixel-level Laplacian (0.097) is worse than the edge filters.
The same pneumonia films are missed every time. About 15 of 190 score below 0.1 in both pass 2 and pass 5. They are crisp, high-contrast films, and they account for most of the remaining error. More contrast features cannot fix them.
Changes for pass 6

To catch a crisp pneumonia film the network needs evidence of opacity itself. Pass 2 had two such channels, but at weights too small to test them.

Opacity channels restored at meaningful weight. Both are read over the lung rows with columns [0.25, 1, 0.7, 0.25].
Channel 9 is grey tissue (0.35–0.5) without edges, weight +2.0. In pass 2 it was 2.4 times higher on pneumonia in the lung cells.
Channel 25 is dense tissue (above 0.5–0.7) without edges, weight +1.5. It was 1.7 times higher on pneumonia.
Their edge coefficients rescaled (0.8 → 0.5 and 0.9 → 0.55), because removing the dead zone raised all edge values by about half.
Bias lowered to +11.6 to offset the new positive terms; this could be off by half a logit.
Aeration probes. Channels 0 and 2 compute edge energy plus 0.15 or 0.30 times the aerated-lung fraction, to see on the same images whether adding aeration improves on channel 24.
Intensity filters restored. conv1 filters 2, 5 and 6 go back to their original roles, and the silhouette channels return as zero-weight monitors.

The contrast terms are unchanged from pass 5. This batch is mostly pass-3 images, where the pure edge readout scored 0.881, so that is the baseline to beat.

Prediction

I expect AUROC of about 0.89 (range 0.87–0.92). A result clearly above 0.881 would mean the opacity channels add something.

python
# weights_diag_v6.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    r2 = math.sqrt(2.0)
    EG = 4.0     # edge gain

    MEAN, D50, D35, D20, BLK, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, EAP, EAN, EBP, EBN, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D50, 0] = -G / 0.50; b1[D50] = 1.0                # relu(1 - m/0.50)
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[D20, 0] = -G / 0.20; b1[D20] = 1.0                # relu(1 - m/0.20)
    w1[BLK, 0] = -G / 0.06; b1[BLK] = 1.0                # relu(1 - m/0.06)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    # edge filters, no dead zone
    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right
    w1[EAP, 0] = EG * (Dy + Dx) / r2                     # "/" edges, brighter bottom-right
    w1[EAN, 0] = -EG * (Dy + Dx) / r2                    # "/" edges, other polarity
    w1[EBP, 0] = EG * (Dy - Dx) / r2                     # "\" edges, brighter bottom-left
    w1[EBN, 0] = -EG * (Dy - Dx) / r2                    # "\" edges, other polarity
    w1[RDG, 0] = EG * (C3 - RING); b1[RDG] = -0.08       # thin bright line / blob

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def capdark(k, coef, pat):
        # (D50 - 0.4*D20)/0.6 : 1 for m <= 0.2, falls to 0 at m = 0.5
        put(k, D50, coef / 0.6, pat)
        put(k, D20, -0.4 * coef / 0.6, pat)

    def air(k, coef, pat):
        # capped darkness that is <= 0 on pure black (padding)
        capdark(k, coef, pat)
        put(k, BLK, -1.5 * coef, pat)

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def edges(k, coef, pat):
        for i in range(7, 15):
            put(k, i, coef, pat)

    # 0, 2: aeration probes - edge energy plus aerated-lung fraction
    edges(0, 0.25, ALL); air(0, 0.15, ALL)
    edges(2, 0.25, ALL); air(2, 0.30, ALL)

    # 1, 4, 7: intensity monitors (zero weight)
    put(1, D50, 1.0, ALL)
    put(4, BLK, 1.0, ALL)                 # padding monitor
    air(7, 1.0, ALL)

    # 8: HAZE50 - homogeneous mid-grey (monitor)
    put(8, D50, -2.5, ALL); put(8, B50, -2.5, ALL); b2[8] = 1.0

    # 9: HAZE_SMOOTH - grey 0.35..0.5 with few edges (edge coef rescaled 0.8 -> 0.5)
    put(9, D35, -5.0, ALL); put(9, B50, -3.0, ALL); edges(9, -0.5, ALL); b2[9] = 1.0

    # 11: vertical-edge energy (monitor)
    put(11, EVR, 1.0, ALL); put(11, EVL, 1.0, ALL)

    # 14: ridge energy
    put(14, RDG, 1.0, ALL)

    # 15-17: single-polarity edges (monitors)
    put(15, EVR, 1.0, ALL)
    put(16, EVL, 1.0, ALL)
    put(17, EHD, 1.0, ALL)

    # 20-22: silhouettes (monitors)
    air(20, 1.0, TOP); dense(20, 1.0, BOT); b2[20] = -1.0       # air above, dense below
    air(21, 1.0, LEFT); dense(21, 1.0, RIGHT); b2[21] = -1.0    # air left, dense right
    dense(22, 1.0, LEFT); air(22, 1.0, RIGHT); b2[22] = -1.0    # dense left, air right

    # 24: total edge energy (main read-out and on-batch reference)
    edges(24, 0.25, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges (edge coef rescaled 0.9 -> 0.55)
    dense(25, 1.0, ALL); edges(25, -0.55, ALL)

    # 26: DENSE_GREY - band 0.5..0.7 (monitor)
    put(26, D50, -5.0, ALL); put(26, B70, -2.0, ALL); b2[26] = 1.0

    # 28-30: coarse brightness gradients (lung-to-mediastinum contrast)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # 31: edge energy away from black padding (reference)
    edges(31, 0.25, ALL); put(31, BLK, -2.0, ALL)

    # channels 3, 5, 6, 10, 12, 13, 18, 19, 23, 27 are unused (all zero) this pass

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral
    C4 = cells([1, 2], [0.0, 1.0, 1.0, 0.0])      # central lung cells only
    DIA = cells([2], [0.5, 1.0, 1.0, 0.5])        # diaphragm row
    M = cells([1, 2], [0.25, 1.0, 0.7, 0.25])     # opacity read-out

    fc = np.zeros((32, 4, 4))

    # fine contrast: rib and vessel edges in the lung rows -> normal
    fc[24] = -5.0 * E4
    fc[14] = -6.0 * E4

    # coarse contrast: dark lung against heart, mediastinum, diaphragm -> normal
    fc[28] = -3.0 * C4
    fc[29] = -3.0 * C4
    fc[30] = -3.0 * DIA

    # opacity: smooth grey or white tissue inside the lung zone -> pneumonia
    fc[9] = 2.0 * M
    fc[25] = 1.5 * M

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([11.6])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 7

What pass 6 showed

Pass 6 scored AUROC 0.903 (accuracy 0.83, sensitivity 0.89, specificity 0.69) on images where the pure edge readout had scored 0.881.

The gain came from removing the edge dead zone, not from the opacity channels. The whole-image edge reference on the same images rose by the same amount (0.904 to 0.924). The opacity channels point the right way (1.6–1.9 times higher on pneumonia in the lung cells), but I cannot yet show they change the ranking.
Aeration adds nothing. Edge energy plus aerated-lung fraction separates no better than edge energy alone (0.077 and 0.091 against 0.076), so that idea is closed.
The bias was too high. The opacity terms added more than I estimated, which is why specificity fell.
The same crisp pneumonia films are still missed (for example P04123, P05165, P02164, P02361).

One thing I should be plain about: the regional probes show pneumonia films have weaker edges even over the arms and chest wall (about 0.8 of normal), not only in the lungs (about 0.6). So part of what this network measures is overall film contrast, which is a property of how these children were imaged, and only part is lung-specific. That may transfer less well to another hospital.

Changes for pass 7
Capped edges. Pass 5 showed that limiting strong edges improves separation (0.060 against 0.071), probably because bones, wires and markers produce large edges unrelated to the lungs. Each horizontal and vertical edge filter now has a twin that fires only above 0.5; conv2 subtracts the twin, so each response is capped at 0.5.
Diagonal edge filters dropped to make room for the twins. All orientations separated equally, so this should cost little.
Slots freed. The near-black, D50 and D20 filters are removed. They fed only monitors, and the padding and aeration questions are settled.
Main edge weight set to −6 on the capped sum, which runs about 20% lower than the old sum.
Opacity channels kept unchanged with their edge term re-expressed in capped edges (coefficients 1.2 and 1.35). I keep them one more round so the next two passes can attribute their effect.
Bias lowered to +10.9 to restore the balance pass 5 had.
Probes.
conv1 filters 13 and 14 give caps at 1.0 and 0.12 for the channel-10 edge. Channels 16–19 then show separation for no cap and caps of 0.12, 0.5 and 1.0 side by side.
conv1 filter 12 is the ridge filter without its dead zone, for comparison with filter 15.
Channel 23 is the uncapped horizontal-plus-vertical sum, as the on-batch reference for channel 24.

Over the next two passes the batches repeat images from passes 5 and 6, which lets me separate the effect of the cap from the effect of the opacity channels.

Prediction

I expect AUROC of about 0.92 (range 0.90–0.94). This batch is mostly pass-4 images, where the earlier model scored 0.918.

python
# weights_diag_v7.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    EG = 4.0     # edge gain
    CAP = 0.5    # edge responses are capped at this value (via twins)

    MEAN, THD, D35, THU, TVR, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, TVL, RDN, TV100, TV12, RDG = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    # edge filters, no dead zone
    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right

    # twins: the part of each edge response above CAP
    w1[THD, 0] = EG * Dy;  b1[THD] = -CAP
    w1[THU, 0] = -EG * Dy; b1[THU] = -CAP
    w1[TVR, 0] = EG * Dx;  b1[TVR] = -CAP
    w1[TVL, 0] = -EG * Dx; b1[TVL] = -CAP

    # probes
    w1[TV100, 0] = -EG * Dx; b1[TV100] = -1.00           # channel 10 above 1.00
    w1[TV12, 0] = -EG * Dx;  b1[TV12] = -0.12            # channel 10 above 0.12
    w1[RDN, 0] = EG * (C3 - RING); b1[RDN] = 0.0         # ridge without dead zone

    w1[RDG, 0] = EG * (C3 - RING); b1[RDG] = -0.08       # thin bright line / blob

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    PAIRS = ((EHD, THD), (EHU, THU), (EVR, TVR), (EVL, TVL))

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def hv(k, coef, pat):
        # capped horizontal + vertical edge energy: sum of min(response, CAP)
        for e, t in PAIRS:
            put(k, e, coef, pat)
            put(k, t, -coef, pat)

    # 9: HAZE_SMOOTH - grey 0.35..0.5 with few edges
    put(9, D35, -5.0, ALL); put(9, B50, -3.0, ALL); hv(9, -1.2, ALL); b2[9] = 1.0

    # 12: smoothed no-dead-zone ridge (probe)
    put(12, RDN, 1.0, ALL)

    # 14: ridge energy
    put(14, RDG, 1.0, ALL)

    # 16-19: cap-level probes on the channel-10 edge
    put(16, EVL, 1.0, ALL)                                   # no cap
    put(17, EVL, 1.0, ALL); put(17, TV12, -1.0, ALL)         # cap 0.12
    put(18, EVL, 1.0, ALL); put(18, TVL, -1.0, ALL)          # cap 0.50
    put(19, EVL, 1.0, ALL); put(19, TV100, -1.0, ALL)        # cap 1.00

    # 23: uncapped horizontal + vertical edge energy (on-batch reference)
    for e, t in PAIRS:
        put(23, e, 0.5, ALL)

    # 24: capped horizontal + vertical edge energy (main read-out)
    hv(24, 0.5, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges
    dense(25, 1.0, ALL); hv(25, -1.35, ALL)

    # 28-30: coarse brightness gradients (lung-to-mediastinum contrast)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # all other conv2 channels are unused (all zero) this pass

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral
    C4 = cells([1, 2], [0.0, 1.0, 1.0, 0.0])      # central lung cells only
    DIA = cells([2], [0.5, 1.0, 1.0, 0.5])        # diaphragm row
    M = cells([1, 2], [0.25, 1.0, 0.7, 0.25])     # opacity read-out

    fc = np.zeros((32, 4, 4))

    # fine contrast: capped rib and vessel edges in the lung rows -> normal
    fc[24] = -6.0 * E4
    fc[14] = -6.0 * E4

    # coarse contrast: dark lung against heart, mediastinum, diaphragm -> normal
    fc[28] = -3.0 * C4
    fc[29] = -3.0 * C4
    fc[30] = -3.0 * DIA

    # opacity: smooth grey or white tissue inside the lung zone -> pneumonia
    fc[9] = 2.0 * M
    fc[25] = 1.5 * M

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([10.9])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 8

What pass 7 showed

Pass 7 scored AUROC 0.924 (accuracy 0.83, sensitivity 0.79, specificity 0.93), against 0.918 from the pass-4 model on mostly the same images.

Capping works on the whole image. The capped horizontal-plus-vertical sum separates at 0.054 against 0.063 uncapped. Across cap levels 0.5 is best: 0.076 at 0.12, 0.063 at 0.5, 0.070 at 1.0 and 0.074 with no cap.
The lung-row readout did not gain from it. My output still trails the uncapped whole-image reference by 0.013, exactly as in pass 4. That gap has been stable for each batch across passes (0.013, about 0.03 and about 0.02), so it is a precise yardstick, and nothing I add inside the lung rows has moved it.
The ridge filter is better without its dead zone (0.093 against 0.122).
The bias was too low. The capped sum came out larger than I estimated, which tipped the balance toward specificity.

My reading is that the fine edges, ridges, coarse gradients and opacity channels are all measuring one underlying quantity, film contrast in the lung zone. That quantity is now measured about as well as this readout allows.

Changes for pass 8
Ridge dead zone removed (filter 15), with its weight eased from −6 to −5 because its values roughly double.
Bias raised to +12.7 to restore the pass-5 balance.
Everything else unchanged, so this pass doubles as the attribution test. Pass 5 ran on mostly these images without capping or opacity channels and scored 0.902, trailing its reference by 0.030. If this pass does not narrow that, I will drop the opacity channels as unproven.
Probes for a new kind of information.
Brightness-relative contrast (channels 3, 4, 5). Capped edges minus 0.15 or 0.30 times local brightness, and plus 0.15. Edge size scales with exposure, so this tests whether correcting for brightness sharpens the measure.
Edges plus ridge (channels 0, 1). Capped edges plus 1.0 or 2.5 times ridge, to see whether ridge adds anything once edges are capped.
Capped ridge and valley. conv1 filters 13 and 14 are ridge twins at 0.25 and 0.12, and filter 12 detects thin dark lines.
Prediction

I expect AUROC of about 0.905 (range 0.89–0.92), roughly level with pass 5 on these images.

python
# weights_diag_v8.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    EG = 4.0     # edge gain
    CAP = 0.5    # edge responses are capped at this value (via twins)

    MEAN, THD, D35, THU, TVR, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, TVL, VAL, TR25, TR12, RDN = 7, 8, 9, 10, 11, 12, 13, 14, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    # edge filters, no dead zone
    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right

    # twins: the part of each edge response above CAP
    w1[THD, 0] = EG * Dy;  b1[THD] = -CAP
    w1[THU, 0] = -EG * Dy; b1[THU] = -CAP
    w1[TVR, 0] = EG * Dx;  b1[TVR] = -CAP
    w1[TVL, 0] = -EG * Dx; b1[TVL] = -CAP

    # ridge (thin bright line), no dead zone
    w1[RDN, 0] = EG * (C3 - RING)

    # probes
    w1[VAL, 0] = -EG * (C3 - RING)                       # thin dark line
    w1[TR25, 0] = EG * (C3 - RING); b1[TR25] = -0.25     # ridge above 0.25
    w1[TR12, 0] = EG * (C3 - RING); b1[TR12] = -0.12     # ridge above 0.12

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    PAIRS = ((EHD, THD), (EHU, THU), (EVR, TVR), (EVL, TVL))

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def hv(k, coef, pat):
        # capped horizontal + vertical edge energy: sum of min(response, CAP)
        for e, t in PAIRS:
            put(k, e, coef, pat)
            put(k, t, -coef, pat)

    # 0, 1: probes - capped edges plus ridge
    hv(0, 0.5, ALL); put(0, RDN, 1.0, ALL)
    hv(1, 0.5, ALL); put(1, RDN, 2.5, ALL)

    # 3, 4, 5: probes - capped edges relative to local brightness
    hv(3, 0.5, ALL); put(3, MEAN, -0.15, ALL)
    hv(4, 0.5, ALL); put(4, MEAN, -0.30, ALL)
    hv(5, 0.5, ALL); put(5, MEAN, 0.15, ALL)

    # 9: HAZE_SMOOTH - grey 0.35..0.5 with few edges
    put(9, D35, -5.0, ALL); put(9, B50, -3.0, ALL); hv(9, -1.2, ALL); b2[9] = 1.0

    # 12: thin dark lines (probe)
    put(12, VAL, 1.0, ALL)

    # 13, 15: capped-ridge probes
    put(13, RDN, 1.0, ALL); put(13, TR25, -1.0, ALL)       # ridge capped at 0.25
    put(15, RDN, 1.0, ALL); put(15, TR12, -1.0, ALL)       # ridge capped at 0.12

    # 14: ridge energy
    put(14, RDN, 1.0, ALL)

    # 23: uncapped horizontal + vertical edge energy (on-batch reference)
    for e, t in PAIRS:
        put(23, e, 0.5, ALL)

    # 24: capped horizontal + vertical edge energy (main read-out)
    hv(24, 0.5, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges
    dense(25, 1.0, ALL); hv(25, -1.35, ALL)

    # 28-30: coarse brightness gradients (lung-to-mediastinum contrast)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # all other conv2 channels are unused (all zero) this pass

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral
    C4 = cells([1, 2], [0.0, 1.0, 1.0, 0.0])      # central lung cells only
    DIA = cells([2], [0.5, 1.0, 1.0, 0.5])        # diaphragm row
    M = cells([1, 2], [0.25, 1.0, 0.7, 0.25])     # opacity read-out

    fc = np.zeros((32, 4, 4))

    # fine contrast: capped rib and vessel edges in the lung rows -> normal
    fc[24] = -6.0 * E4
    fc[14] = -5.0 * E4

    # coarse contrast: dark lung against heart, mediastinum, diaphragm -> normal
    fc[28] = -3.0 * C4
    fc[29] = -3.0 * C4
    fc[30] = -3.0 * DIA

    # opacity: smooth grey or white tissue inside the lung zone -> pneumonia
    fc[9] = 2.0 * M
    fc[25] = 1.5 * M

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([12.7])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 9

One note first: the harness labelled that run “PRACTICE PASS 1/20” and reused the first 256 images, so its rotation appears to have restarted. I am counting it as my eighth pass.

What pass 8 showed

Pass 8 scored AUROC 0.930 (accuracy 0.89, sensitivity 0.88, specificity 0.92). The bias is now well balanced.

The gap has not moved. My output trails the uncapped whole-image reference by 0.014 on these images (0.930 against 0.944). It was 0.013 in passes 4 and 7 on essentially the same images. Removing the dead zone raised output and reference together, and nothing since has closed the distance.
Ridge dilutes the capped edges. On the whole image, capped edges alone separate at 0.046. Adding ridge makes it 0.053 at weight 1.0 and 0.058 at 2.5.
Brightness correction does not help. Subtracting local brightness gives 0.056 and 0.085, and adding it gives 0.050; all are worse than 0.046.
Capped ridge and valley are no better. They separate at 0.064–0.070, behind the capped edges.

This run was on the first batch, not the pass-5 batch, so it could not test the opacity channels as I intended. That question is still open.

Because ridge is a worse copy of the edge measure, the coarse-gradient and opacity terms may be diluting it too. They now make up more than half of the logit’s class difference.

Changes for pass 9
Pure readout. The logit is the capped edge sum (channel 24) on the lung mask at weight −12, with bias +15.3, and nothing else. If this matches or beats the combined model against the on-batch reference, the extra terms are not earning their place and the final network should be simpler.
Ridge dropped, on the pass-8 evidence.
A full 4×4 map of the signal. Channels 0–15 are copies of the capped edge sum, each read in one grid cell at a weight of −0.05. Their logit columns give the class means for every cell, which I will use to set the final mask.
Coarse and opacity channels kept as zero-weight monitors, so their separations stay visible.
Prediction

I expect AUROC of about 0.90 on the second batch, or about 0.93 if the harness serves the first batch again (range 0.88–0.94). What matters is the distance to channel 23’s separation on the same images.

python
# weights_diag_v9.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: 16 hand-designed 5x5 filters
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    G = np.outer(g, g)                                   # local mean (sums to 1)
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0]) / 4.0
    C3 = np.outer(h, h)                                  # centre 3x3 blob (sums to 1)
    RING = np.ones((5, 5)) / 16.0                        # outer ring (sums to 1)
    RING[1:4, 1:4] = 0.0
    EG = 4.0     # edge gain
    CAP = 0.5    # edge responses are capped at this value (via twins)

    MEAN, THD, D35, THU, TVR, B50, B70 = 0, 1, 2, 3, 4, 5, 6
    EHD, EHU, EVR, EVL, TVL, RDN = 7, 8, 9, 10, 11, 15

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)

    w1[MEAN, 0] = G                                      # m
    w1[D35, 0] = -G / 0.35; b1[D35] = 1.0                # relu(1 - m/0.35)
    w1[B50, 0] = G / 0.50;  b1[B50] = -1.0               # relu((m-0.5)/0.5)
    w1[B70, 0] = G / 0.30;  b1[B70] = -0.70 / 0.30       # relu((m-0.7)/0.3)

    # edge filters, no dead zone
    w1[EHD, 0] = EG * Dy                                 # dark above, bright below
    w1[EHU, 0] = -EG * Dy                                # bright above, dark below
    w1[EVR, 0] = EG * Dx                                 # dark left, bright right
    w1[EVL, 0] = -EG * Dx                                # bright left, dark right

    # twins: the part of each edge response above CAP
    w1[THD, 0] = EG * Dy;  b1[THD] = -CAP
    w1[THU, 0] = -EG * Dy; b1[THU] = -CAP
    w1[TVR, 0] = EG * Dx;  b1[TVR] = -CAP
    w1[TVL, 0] = -EG * Dx; b1[TVL] = -CAP

    # ridge (thin bright line), no dead zone - monitor only
    w1[RDN, 0] = EG * (C3 - RING)

    # conv1 filters 12, 13, 14 are unused (all zero) this pass

    # ------------------------------------------------------------------
    # conv2: 32 channels built from 3x3 spatial patterns
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)
    TOP = np.zeros((3, 3)); TOP[0, :] = 1.0 / 3.0
    BOT = np.zeros((3, 3)); BOT[2, :] = 1.0 / 3.0
    LEFT = np.zeros((3, 3)); LEFT[:, 0] = 1.0 / 3.0
    RIGHT = np.zeros((3, 3)); RIGHT[:, 2] = 1.0 / 3.0

    PAIRS = ((EHD, THD), (EHU, THU), (EVR, TVR), (EVL, TVL))

    def put(k, i, coef, pat):
        w2[k, i] += coef * pat

    def dense(k, coef, pat):
        # (B50 - 0.6*B70)/0.4 : 0 for m <= 0.5, 1 for m >= 0.7
        put(k, B50, coef / 0.4, pat)
        put(k, B70, -1.5 * coef, pat)

    def hv(k, coef, pat):
        # capped horizontal + vertical edge energy: sum of min(response, CAP)
        for e, t in PAIRS:
            put(k, e, coef, pat)
            put(k, t, -coef, pat)

    # 0-15: copies of the capped edge sum, one per grid cell (regional map)
    for k in range(16):
        hv(k, 0.5, ALL)

    # 17: HAZE_SMOOTH - grey 0.35..0.5 with few edges (monitor)
    put(17, D35, -5.0, ALL); put(17, B50, -3.0, ALL); hv(17, -1.2, ALL); b2[17] = 1.0

    # 18: ridge energy (monitor)
    put(18, RDN, 1.0, ALL)

    # 23: uncapped horizontal + vertical edge energy (on-batch reference)
    for e, t in PAIRS:
        put(23, e, 0.5, ALL)

    # 24: capped horizontal + vertical edge energy (the only real read-out)
    hv(24, 0.5, ALL)

    # 25: BRIGHT_SMOOTH - dense without edges (monitor)
    dense(25, 1.0, ALL); hv(25, -1.35, ALL)

    # 28-30: coarse brightness gradients (monitors)
    put(28, MEAN, 3.0, RIGHT); put(28, MEAN, -3.0, LEFT)   # brighter right
    put(29, MEAN, 3.0, LEFT);  put(29, MEAN, -3.0, RIGHT)  # brighter left
    put(30, MEAN, 3.0, BOT);   put(30, MEAN, -3.0, TOP)    # brighter below

    # all other conv2 channels are unused (all zero) this pass

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral

    fc = np.zeros((32, 4, 4))

    # the single real term: capped edge energy in the lung rows -> normal
    fc[24] = -12.0 * E4

    # regional map: channel k reads only grid cell k (row k // 4, column k % 4)
    for k in range(16):
        fc[k, k // 4, k % 4] = -0.05

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([15.3])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

PRACTICE PASS 10

What pass 9 showed

The pure capped-edge readout scored AUROC 0.916 (accuracy 0.87, sensitivity 0.84, specificity 0.95) on exactly the pass-2 images. The harness is still on its restarted count; this is my ninth pass.

Simpler is better. On these images the combined models scored 0.899 (pass 2) and 0.902 (pass 5). The gap to the uncapped whole-image reference fell from 0.026–0.030 to 0.019. The ridge, coarse-gradient and opacity terms were diluting the main signal, so I am dropping all of them.
The 4×4 map confirms the mask. Pneumonia capped-edge energy as a fraction of normal, cell by cell:
	col 0	col 1	col 2	col 3
row 0	0.62	0.45	0.46	0.56
row 1	0.90	0.71	0.72	0.89
row 2	0.87	0.67	0.63	0.84
row 3	0.59	0.52	0.46	0.55

The four central lung cells carry a real difference, and the lateral lung cells carry almost none. Rows 0 and 3 show the largest gaps, but those are the padded rows, where pneumonia films simply have less picture. I am still not using them. On the practice batches so far, that choice costs about 0.02 to 0.03 of AUROC against the padding-cleaned whole-image figure.

Changes for pass 10
Same pure model, to get its baseline on the third batch. I then have a like-for-like reference on each batch for testing any further change.
Regional probe weights removed, so the readout is strictly the lung rows.
Bias raised to +15.7. Pass 9 leaned toward specificity (0.95 against 0.84 sensitivity).
Cap-level test on the full edge sum. With the dropped channels gone, conv1 has room for two more sets of twins. Channels 20 and 21 are the edge sum capped at 0.3 and 0.8, beside 0.5 in channel 24 and no cap in channel 23. Each also gets a weight of −0.05 on the lung mask, so I see lung-cell class means as well as whole-image separation.

The network is now four edge filters with their caps, one conv2 channel, and one mask.

Prediction

I expect AUROC of about 0.91 on the third batch (range 0.89–0.93). The combined model scored 0.903 there.

python
# weights_diag_v10.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: four edge filters plus three sets of cap twins
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    EG = 4.0     # edge gain
    CAP = 0.5    # main cap on each edge response
    CAP_LO = 0.3 # probe cap
    CAP_HI = 0.8 # probe cap

    # kernels of the four edge directions
    KERN = (EG * Dy,      # dark above, bright below
            -EG * Dy,     # bright above, dark below
            EG * Dx,      # dark left, bright right
            -EG * Dx)     # bright left, dark right

    EDGE = (0, 1, 2, 3)          # plain edge responses, no dead zone
    TWIN = (4, 5, 6, 7)          # part of each response above CAP
    TWIN_LO = (8, 9, 10, 11)     # part above CAP_LO (probe)
    TWIN_HI = (12, 13, 14, 15)   # part above CAP_HI (probe)

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)
    for j in range(4):
        w1[EDGE[j], 0] = KERN[j]
        w1[TWIN[j], 0] = KERN[j];    b1[TWIN[j]] = -CAP
        w1[TWIN_LO[j], 0] = KERN[j]; b1[TWIN_LO[j]] = -CAP_LO
        w1[TWIN_HI[j], 0] = KERN[j]; b1[TWIN_HI[j]] = -CAP_HI

    # ------------------------------------------------------------------
    # conv2: 3x3 averages of capped edge energy
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)

    def capped(k, twins):
        # 0.5 * sum over the four directions of min(response, cap)
        for j in range(4):
            w2[k, EDGE[j]] += 0.5 * ALL
            w2[k, twins[j]] += -0.5 * ALL

    # 20, 21: probe caps
    capped(20, TWIN_LO)      # capped at 0.3
    capped(21, TWIN_HI)      # capped at 0.8

    # 23: uncapped edge energy (on-batch reference)
    for j in range(4):
        w2[23, EDGE[j]] += 0.5 * ALL

    # 24: edge energy capped at 0.5 (the only real read-out)
    capped(24, TWIN)

    # all other conv2 channels are unused (all zero)

    # ------------------------------------------------------------------
    # fc: per-channel 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral

    fc = np.zeros((32, 4, 4))

    # capped edge energy in the lung rows -> normal
    fc[24] = -12.0 * E4

    # probe caps read on the same mask (negligible weight)
    fc[20] = -0.05 * E4
    fc[21] = -0.05 * E4

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([15.7])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Over 90% of your 5-hour limit used · Resets at 10:00 PMGet more usage
Pasted text.txt
TXT

PRACTICE PASS 11

What pass 10 showed

The pure capped-edge model scored AUROC 0.902 (accuracy 0.84, sensitivity 0.85, specificity 0.81) on exactly the pass-3 images. This is my tenth pass.

Pure is at least as good as combined here too. The combined model scored 0.903 on mostly the same images in pass 6, and the original dead-zone readout scored 0.881 in pass 3. With pass 9 (0.916 pure against 0.902 combined), the simpler model is equal or better on both batches where I can compare, so I am keeping it.
Cap level is close to flat. On the whole image the edge sum separates at 0.058 with a cap of 0.3, 0.063 at 0.5, 0.070 at 0.8 and 0.080 uncapped. A lower cap looks slightly better here, but the earlier single-filter test showed 0.24 slightly worse than 0.5, and I have no lung-row evidence either way. I am leaving the cap at 0.5.
The bias sits in the right range. Pass 9 leaned toward specificity at +15.3, and this pass leaned slightly toward sensitivity at +15.7, on different batches.

I also checked the padded rows against pass 9’s map. In the top row pneumonia films show about 45% of the normal edge energy, which is what their smaller share of real picture in that row would produce on its own. So the top and bottom rows contain no pathology signal, only the padding shortcut, and leaving them out costs nothing real.

Changes for pass 11
Bias set to +15.5, midway between the last two.
Probe twins and their channels removed. conv1 is now four edge filters and four cap twins; conv2 has the capped sum (channel 24) and the uncapped reference (channel 23, zero weight).
Nothing else. This is a dress rehearsal of the network I intend to submit, run on the remaining batch so I have its score and balance on all three.

If this pass is in line with the other two, I will submit this network as FINAL with the unused reference channel removed, which does not change its output.

Prediction

I expect AUROC of about 0.93 on the first batch (range 0.91–0.94), where the combined model scored 0.930.

python
# weights_diag_v11.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: four edge filters and their cap twins
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    EG = 4.0     # edge gain
    CAP = 0.5    # cap on each edge response

    # kernels of the four edge directions
    KERN = (EG * Dy,      # dark above, bright below
            -EG * Dy,     # bright above, dark below
            EG * Dx,      # dark left, bright right
            -EG * Dx)     # bright left, dark right

    EDGE = (0, 1, 2, 3)          # plain edge responses, no dead zone
    TWIN = (4, 5, 6, 7)          # part of each response above CAP

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)
    for j in range(4):
        w1[EDGE[j], 0] = KERN[j]
        w1[TWIN[j], 0] = KERN[j]; b1[TWIN[j]] = -CAP

    # conv1 filters 8-15 are unused (all zero)

    # ------------------------------------------------------------------
    # conv2: 3x3 average of capped edge energy
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)

    # 23: uncapped edge energy (on-batch reference, zero fc weight)
    for j in range(4):
        w2[23, EDGE[j]] += 0.5 * ALL

    # 24: edge energy capped at CAP: 0.5 * sum over directions of min(response, CAP)
    for j in range(4):
        w2[24, EDGE[j]] += 0.5 * ALL
        w2[24, TWIN[j]] += -0.5 * ALL

    # all other conv2 channels are unused (all zero)

    # ------------------------------------------------------------------
    # fc: 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral

    fc = np.zeros((32, 4, 4))

    # capped edge energy in the lung rows -> normal
    fc[24] = -12.0 * E4

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([15.5])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),
    }
Pasted text.txt
TXT

FINAL

This is the pass-11 network with the zero-weight reference channel removed, which does not change its output. Across the three practice batches it scored AUROC 0.926, 0.916 and 0.902, with sensitivity and specificity both near 0.85.

What I saw in the examples

Normal lungs are dark with crisp ribs and sharp heart and diaphragm borders. Pneumonia lungs are hazy grey with faint ribs and blurred borders. The lungs sit in grid rows 1–2, mostly in columns 1–2. Pneumonia films also tend to be wider, so they carry more black padding top and bottom; I treated that as an acquisition artefact and designed around it.

The network
conv1 (8 of 16 filters used).
Filters 0–3 are smoothed 5×5 derivative filters with gain 4 and no dead zone: brighter-below, brighter-above, brighter-right and brighter-left.
Filters 4–7 are twins of the same four kernels with bias −0.5, so each fires only on the part of the response above 0.5.
Filters 8–15 are zero.
conv2 (1 of 32 channels used). Channel 24 is the 3×3 average of half the sum, over the four directions, of each edge response minus its twin. That is each response capped at 0.5, so bones, wires and markers cannot dominate.
Final layer. Channel 24 has weight −12 in rows 1–2 with column weights [0.15, 1, 1, 0.15]: more edge energy in the central lung cells means normal. Every other weight is zero. Rows 0 and 3 are excluded because their class difference is fully explained by padding.
Bias. +15.5 puts the decision point at a lung-cell edge level between the two classes, chosen to balance sensitivity and specificity across the three practice batches.
Passes used

The harness restarted its own count after my seventh pass; this is my numbering.

pass	AUROC	what I changed and why
1	0.773	First design: intensity, haze, edge and silhouette channels with prior weights.
2	0.899	Diagnostics showed edge channels separate strongly and lung brightness not at all, so I made edge energy the main vote, zeroed flat channels and reset the bias.
3	0.881	Stripped to edge energy alone, as a controlled test with regional and padding probes. The extras had added nothing, and whole-image edges looked padding-driven.
4	0.918	Down-weighted lateral columns, added coarse gradients and ridge, and tested edge-filter variants.
5	0.902	Removed the edge dead zone (probe 0.068 against 0.076) and tightened the coarse masks. A cleaner test confirmed the whole-image advantage is the padding shortcut.
6	0.903	Restored two opacity channels at meaningful weight to target crisp pneumonia films; no measurable gain.
7	0.924	Capped edge responses at 0.5 (probe 0.060 against 0.071) and dropped the diagonal filters to make room.
8	0.930	Removed the ridge dead zone and fixed the bias. Probes showed ridge and brightness correction both dilute the capped edges.
9	0.916	Dropped ridge, coarse and opacity terms. Pure capped edges beat the combined model on the same images (0.916 against 0.899–0.902).
10	0.902	Same pure model on the third batch (combined had scored 0.903); tested cap levels and kept 0.5.
11	0.926	Dress rehearsal with bias +15.5 and all probes removed.

Kept because they helped: the central lung mask, no dead zone, the 0.5 cap, and removing everything else. Rejected as unhelpful or diluting: intensity and haze channels, aeration, silhouettes, ridge, coarse gradients, opacity and brightness correction.

What this network does and does not measure
It measures film contrast in the lung zone. That is partly pathology (hazy lungs hide ribs) and partly how these children were imaged. Pneumonia films here have weaker edges even over the arms and chest wall.
It misses crisp pneumonia films. About 1 in 10 pneumonia films in practice are high-contrast and score as normal. No opacity feature I built rescued them.
It does not use the padding shortcut. On the practice batches, using the whole frame would have added roughly 0.02 to 0.03 AUROC from padding alone. A trained network will likely take that.
Prediction
Same-hospital held-out set: AUROC about 0.91 (range 0.89–0.93), accuracy about 0.85.
Different hospital: AUROC lower and uncertain, my guess 0.75–0.88. Accuracy at P = 0.5 may be poor there, because the threshold is an absolute edge level and another hospital’s images may be systematically sharper or softer.
python
# weights_diag_final.py
import numpy as np
import math


def make_weights():
    # ------------------------------------------------------------------
    # conv1: four edge filters and their cap twins
    # ------------------------------------------------------------------
    g = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0      # binomial smoothing
    d = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / 3.0     # derivative, unit step -> 1
    Dy = np.outer(d, g)                                  # + when brighter BELOW
    Dx = np.outer(g, d)                                  # + when brighter to the RIGHT
    EG = 4.0     # edge gain
    CAP = 0.5    # cap on each edge response

    # kernels of the four edge directions
    KERN = (EG * Dy,      # dark above, bright below
            -EG * Dy,     # bright above, dark below
            EG * Dx,      # dark left, bright right
            -EG * Dx)     # bright left, dark right

    EDGE = (0, 1, 2, 3)          # plain edge responses, no dead zone
    TWIN = (4, 5, 6, 7)          # part of each response above CAP

    w1 = np.zeros((16, 1, 5, 5))
    b1 = np.zeros(16)
    for j in range(4):
        w1[EDGE[j], 0] = KERN[j]
        w1[TWIN[j], 0] = KERN[j]; b1[TWIN[j]] = -CAP

    # conv1 filters 8-15 are unused (all zero)

    # ------------------------------------------------------------------
    # conv2: 3x3 average of capped edge energy
    # ------------------------------------------------------------------
    w2 = np.zeros((32, 16, 3, 3))
    b2 = np.zeros(32)

    ALL = np.full((3, 3), 1.0 / 9.0)

    # 24: edge energy capped at CAP: 0.5 * sum over directions of min(response, CAP)
    for j in range(4):
        w2[24, EDGE[j]] += 0.5 * ALL
        w2[24, TWIN[j]] += -0.5 * ALL

    # all other conv2 channels are unused (all zero)

    # ------------------------------------------------------------------
    # fc: 4x4 cell weights (row 0 = top, col 0 = image left)
    # ------------------------------------------------------------------
    def cells(rr, cols, scale=1.0):
        m = np.zeros((4, 4))
        for r in rr:
            m[r, :] = scale * np.array(cols)
        return m

    E4 = cells([1, 2], [0.15, 1.0, 1.0, 0.15])    # central lung cells, light lateral

    fc = np.zeros((32, 4, 4))

    # capped edge energy in the lung rows -> normal
    fc[24] = -12.0 * E4

    fc_w = fc.reshape(1, 512)
    fc_b = np.array([15.5])

    return {
        "conv1.weight": w1.astype(np.float32),
        "conv1.bias": b1.astype(np.float32),
        "conv2.weight": w2.astype(np.float32),
        "conv2.bias": b2.astype(np.float32),
        "fc.weight": fc_w.astype(np.float32),
        "fc.bias": fc_b.astype(np.float32),