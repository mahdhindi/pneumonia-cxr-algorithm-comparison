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