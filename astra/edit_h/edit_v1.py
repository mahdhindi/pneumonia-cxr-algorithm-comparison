# edit_v1.py
# PRACTICE PASS 1

def edit(weight, bias):
    """Return edited weights and bias without training or fitting."""
    new_weight = weight.copy()
    new_bias = bias

    # Strong pneumonia-associated features: add a fixed positive vote.
    pneumonia_features = [
        1988, 1264, 22, 1487,
        197, 260, 1570, 1457,
        1842, 622, 1898, 69,
        768, 446, 149, 596,
    ]

    # Strong normal-associated features: add a fixed negative vote.
    normal_features = [
        1249, 384, 573, 675,
        155, 965, 669, 1696,
        296, 1250, 1309, 1418,
        455, 1399, 1869, 1078,
    ]

    new_weight[pneumonia_features] += 0.25
    new_weight[normal_features] -= 0.25

    return new_weight, new_bias