# edit_v7.py
# PRACTICE PASS 7

def edit(weight, bias):
    """Return a hand-designed compact head without training or fitting."""
    new_weight = weight.copy()
    new_weight[:] = 0.0

    # New: require a fixed two-unit margin of pneumonia evidence.
    new_bias = -2.0

    # Retain all pneumonia-associated features from Pass 6.
    pneumonia_features = [
        1988, 1264, 22, 1487,
        197, 260, 1570, 1457,
        1842, 622, 1898, 69,
        768, 446, 149, 596,
        1995, 984, 535, 1889,
        869, 1708, 1837, 941,
    ]

    # Retain all normal-associated features from Pass 6.
    normal_features = [
        1249, 384, 573, 675,
        155, 965, 669, 1696,
        296, 1250, 1309, 1418,
        455, 1399, 1869, 1078,
        1953, 1707, 1443, 547, 1423,
        1823, 1281, 102, 460,
    ]

    new_weight[pneumonia_features] = 1.0
    new_weight[normal_features] = -1.0

    return new_weight, new_bias