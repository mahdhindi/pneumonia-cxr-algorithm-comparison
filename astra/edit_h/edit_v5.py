# edit_v5.py
# PRACTICE PASS 5

def edit(weight, bias):
    """Return a hand-designed compact head without training or fitting."""
    new_weight = weight.copy()
    new_weight[:] = 0.0
    new_bias = 0.0

    # Retain the pneumonia-associated features from Pass 4.
    pneumonia_features = [
        1988, 1264, 22, 1487,
        197, 260, 1570, 1457,
        1842, 622, 1898, 69,
        768, 446, 149, 596,
    ]

    # Retain the normal-associated features from Pass 4.
    normal_features = [
        1249, 384, 573, 675,
        155, 965, 669, 1696,
        296, 1250, 1309, 1418,
        455, 1399, 1869, 1078,
        1953, 1707, 1443, 547, 1423,
    ]

    new_weight[pneumonia_features] = 1.0
    new_weight[normal_features] = -1.0

    # New: strong separation with low average normal activation.
    additional_pneumonia_features = [
        1995, 984, 535, 1889,
        869, 1708, 1837, 941,
    ]
    new_weight[additional_pneumonia_features] = 1.0

    return new_weight, new_bias