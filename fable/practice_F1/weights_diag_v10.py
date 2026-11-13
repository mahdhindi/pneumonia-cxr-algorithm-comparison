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