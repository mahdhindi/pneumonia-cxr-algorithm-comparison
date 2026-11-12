# edit_v4.py
# PRACTICE PASS 4

def edit(weight, bias):
    """Return a hand-designed compact head without training or fitting."""
    new_weight = weight.copy()
    new_weight[:] = 0.0
    new_bias = 0.0

    # Strong pneumonia-associated features selected in Pass 1.
    pneumonia_features = [
        1988, 1264, 22, 1487,
        197, 260, 1570, 1457,
        1842, 622, 1898, 69,
        768, 446, 149, 596,
    ]

    # Strong normal-associated features from Pass 1, plus the
    # rarely silent normal-associated features retained in Pass 2.
    normal_features = [
        1249, 384, 573, 675,
        155, 965, 669, 1696,
        296, 1250, 1309, 1418,
        455, 1399, 1869, 1078,
        1953, 1707, 1443, 547, 1423,
    ]

    new_weight[pneumonia_features] = 1.0
    new_weight[normal_features] = -1.0

    return new_weight, new_bias