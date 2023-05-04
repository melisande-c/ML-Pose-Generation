from collections import OrderedDict
from torch import nn


class VGGMod(nn.Module):
    def __init__(self, vgg_model, dim, n_features):
        super(VGGMod, self).__init__()
        self.dim = dim
        self.n_features = n_features
        self.vgg_model = vgg_model
        in_features = self.vgg_model.classifier[-4].out_features
        self.vgg_model.classifier[-1] = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(in_features, (64**2) * n_features)),
                    ("upsample", nn.Upsample((dim, dim))),
                ]
            )
        )

    def forward(self, x):
        y = self.vgg_model(x)
        return y.view(self.dim, self.dim, self.n_features)
