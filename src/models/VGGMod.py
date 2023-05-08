from collections import OrderedDict
from torch import nn


class VGGCoordMod(nn.Module):
    def __init__(self, vgg_model, coords_n) -> None:
        super().__init__()
        self.coords_n = coords_n
        self.vgg_model = vgg_model
        in_features = self.vgg_model.classifier[-4].out_features
        self.vgg_model.classifier[-1] = nn.Linear(in_features, 2 * coords_n)

    def forward(self, x):
        y = self.vgg_model(x)
        return nn.Unflatten(1, (self.coords_n, 2))


class VGGSpatialMod(nn.Module):
    def __init__(self, vgg_model, output_dims, n_features):
        super(VGGSpatialMod, self).__init__()
        self.dim = output_dims
        self.n_features = n_features
        self.vgg_model = vgg_model
        in_features = self.vgg_model.classifier[-4].out_features
        self.vgg_model.classifier[-1] = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(in_features, (64**2) * n_features)),
                    ("reshape", nn.Unflatten(1, (n_features, 64, 64))),
                    ("upsample", nn.Upsample((output_dims, output_dims))),
                ]
            )
        )

    def forward(self, x):
        y = self.vgg_model(x)
        return y
