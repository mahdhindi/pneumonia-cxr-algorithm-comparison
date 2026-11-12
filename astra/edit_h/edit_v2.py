# edit_v2.py
# PRACTICE PASS 2

def edit(weight, bias):
    """Return edited weights and bias without training or fitting."""
    new_weight = weight.copy()
    new_bias = bias

    # Retain the successful Pass 1 edits.
    pneumonia_features = [
        1988, 1264, 22, 1487,
        197, 260, 1570, 1457,
        1842, 622, 1898, 69,
        768, 446, 149, 596,
    ]

    normal_features = [
        1249, 384, 573, 675,
        155, 965, 669, 1696,
        296, 1250, 1309, 1418,
        455, 1399, 1869, 1078,
    ]

    new_weight[pneumonia_features] += 0.25
    new_weight[normal_features] -= 0.25

    # New: strengthen normal-associated features that are rarely silent.
    additional_normal_features = [1953, 1707, 1443, 547, 1423]
    new_weight[additional_normal_features] -= 0.25

    return new_weight, new_bias