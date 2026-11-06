"""The five algorithms compared.  All emit one logit (P(pneumonia))."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MajorityClass(nn.Module):
    """Trivial reference: always predict the training-set majority class.
    Has no trainable parameters; `fit` stores the training prior as a logit."""

    def __init__(self):
        super().__init__()
        self.register_buffer("logit", torch.zeros(1))

    def fit(self, prior_pos: float):
        p = min(max(prior_pos, 1e-6), 1 - 1e-6)
        self.logit.fill_(float(torch.logit(torch.tensor(p))))

    def forward(self, x):
        return self.logit.expand(x.shape[0], 1)


class LogisticRegression(nn.Module):
    """The required simple baseline: a single linear layer on the flattened,
    normalised pixels (= multinomial/binary logistic regression trained by
    gradient descent).  Permutation-invariant w.r.t. pixel positions."""

    def __init__(self, img_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * img_size * img_size, 1))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Compact CNN trained from scratch (4 conv blocks, ~0.4M params)."""

    def __init__(self, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in widths:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x))


class WrittenCNN(nn.Module):
    """Project experiment ("weight-space programming", after Takehi et al. 2026):
    a tiny CNN whose weights are WRITTEN by an LLM with no training, and which
    is also trained normally as the comparison ("writtencnn" in ZOO).

    Input: the standard pipeline tensor [B, 3, 224, 224] (ImageNet-normalised
    copies of one grayscale image).  The first thing forward() does is undo that,
    so the layers see ONE channel with values in [0, 1] (0 = black, 1 = white).

        g      [B, 1, 224, 224]  grayscale in [0, 1]
        conv1  Conv2d(1 -> 16, 5x5, stride 2, padding 2) -> ReLU -> MaxPool 2   => [B, 16, 56, 56]
        conv2  Conv2d(16 -> 32, 3x3, stride 1, padding 1) -> ReLU -> MaxPool 2  => [B, 32, 28, 28]
        pool   AdaptiveAvgPool2d(4)  (a 4 x 4 grid over the image)              => [B, 32, 4, 4]
        fc     flatten (channel-major: index = c*16 + row*4 + col) -> Linear(512 -> 1) = logit
        P(pneumonia) = sigmoid(logit)

    5,569 parameters: conv1.weight [16,1,5,5], conv1.bias [16], conv2.weight [32,16,3,3],
    conv2.bias [32], fc.weight [1,512], fc.bias [1]."""

    SHAPES = {"conv1.weight": (16, 1, 5, 5), "conv1.bias": (16,), "conv2.weight": (32, 16, 3, 3),
              "conv2.bias": (32,), "fc.weight": (1, 512), "fc.bias": (1,)}

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        g = x[:, :1] * 0.229 + 0.485                      # undo ImageNet normalisation of channel 0 -> [0, 1]
        h = F.max_pool2d(F.relu(self.conv1(g)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        return self.fc(torch.flatten(self.pool(h), 1))


def resnet50(pretrained: bool = True):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = torchvision.models.resnet50(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def densenet121(pretrained: bool = True):
    w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.densenet121(weights=w)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    return m


def vit_deit_b(pretrained: bool = True, img_size: int = 224):
    """Paper-critique reality check (Assignment 2): the model of Singh et al.,
    Sci. Rep. 14:2487 (2024) - DeiT-Base/16 at 224 px with ImageNet-1k
    weights ('DEIT_Base_Patch16_224'), fully fine-tuned, one-logit head.
    Weights come from timm (facebook/deit_base_patch16_224)."""
    import timm  # imported lazily so the A1 models keep working without timm
    # img_size is 224 in every real run; other sizes (smoke test) make timm interpolate the position embeddings
    return timm.create_model("deit_base_patch16_224", pretrained=pretrained, num_classes=1, img_size=img_size)


# name -> default learning rate, needs training?, family label
# A1 rule: only the LR differs between models (from-scratch 1e-3, pretrained 1e-4).
# vit_deit_b uses the LR reported by Singh et al. (1e-5) so the re-run is faithful
# to the paper; scripts/run_a2.sh also runs it at the A1 pretrained LR (1e-4).
ZOO = {
    "majority":    dict(lr=0.0,  trainable=False, pretrained=False, family="baseline (no learning)"),
    "logreg":      dict(lr=1e-4, trainable=True,  pretrained=False, family="linear baseline"),
    "smallcnn":    dict(lr=1e-3, trainable=True,  pretrained=False, family="CNN from scratch"),
    "writtencnn":  dict(lr=1e-3, trainable=True,  pretrained=False, family="tiny CNN (5.6k params), trained - comparison for LLM-written weights"),
    "resnet50":    dict(lr=1e-4, trainable=True,  pretrained=True,  family="CNN, ImageNet-pretrained, fine-tuned"),
    "densenet121": dict(lr=1e-4, trainable=True,  pretrained=True,  family="CNN, ImageNet-pretrained, fine-tuned"),
    "vit_deit_b":  dict(lr=1e-5, trainable=True,  pretrained=True,  family="ViT (DeiT-B/16), ImageNet-pretrained, fine-tuned"),
}


def build_model(name: str, img_size: int, pretrained: bool = True) -> nn.Module:
    if name == "majority":
        return MajorityClass()
    if name == "logreg":
        return LogisticRegression(img_size)
    if name == "smallcnn":
        return SmallCNN()
    if name == "writtencnn":
        return WrittenCNN()
    if name == "resnet50":
        return resnet50(pretrained)
    if name == "densenet121":
        return densenet121(pretrained)
    if name == "vit_deit_b":
        return vit_deit_b(pretrained, img_size)
    raise ValueError(f"unknown model {name}; choose from {list(ZOO)}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
