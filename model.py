"""SFCN architecture compatible with the supplied V3 checkpoint."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _block(cin, cout, pool, k, p):
    layers = [nn.Conv3d(cin, cout, k, padding=p), nn.BatchNorm3d(cout)]
    if pool:
        layers.append(nn.MaxPool3d(2, stride=2))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class SFCN(nn.Module):
    CH = (32, 64, 128, 256, 256, 64)

    def __init__(self, n_bins=64, dropout=0.5):
        super().__init__()
        fe, cin = nn.Sequential(), 1
        for i, c in enumerate(self.CH):
            last = i == len(self.CH) - 1
            fe.add_module(f"conv_{i}", _block(cin, c, pool=not last,
                                               k=1 if last else 3,
                                               p=0 if last else 1))
            cin = c
        self.feature_extractor = fe
        cl = nn.Sequential()
        cl.add_module("average_pool", nn.AvgPool3d((5, 6, 5)))
        cl.add_module("dropout", nn.Dropout(dropout))
        cl.add_module("conv_6", nn.Conv3d(cin, n_bins, 1))
        self.classifier = cl

    def forward(self, x):
        x = self.classifier(self.feature_extractor(x))
        return F.log_softmax(x.flatten(1), dim=1)
