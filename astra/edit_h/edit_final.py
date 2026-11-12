# edit_final.py
# FINAL — identical to Practice Pass 7

def edit(weight, bias):
    """Return the hand-designed final layer without training or fitting."""
    new_weight = weight.copy()
    new_weight[:] = 0.0
    new_bias = -2.0

    pneumonia_features = [
        # Selected in Pass 1; assigned unit weights in Pass 4.
        1988, 1264, 22, 1487,
        197, 260, 1570, 1457,
        1842, 622, 1898, 69,
        768, 446, 149, 596,

        # Added in Pass 5: strong separation, low normal activation.
        1995, 984, 535, 1889,
        869, 1708, 1837, 941,
    ]

    normal_features = [
        # Selected in Pass 1; assigned unit weights in Pass 4.
        1249, 384, 573, 675,
        155, 965, 669, 1696,
        296, 1250, 1309, 1418,
        455, 1399, 1869, 1078,

        # Selected in Pass 2: rarely silent normal evidence.
        1953, 1707, 1443, 547, 1423,

        # Added in Pass 6: strong normal evidence, moderate silence.
        1823, 1281, 102, 460,
    ]

    new_weight[pneumonia_features] = 1.0
    new_weight[normal_features] = -1.0

    return new_weight, new_bias