from torch import nn


class VGGMod(nn.Module):
    def __init__(self, vgg_model, dim, n_features):
        super(VGGMod, self).__init__()
        self.dim = dim
        self.n_features = n_features
        self.vgg_model = vgg_model
        in_features = self.vgg_model.classifier[-4].out_features
        self.vgg_model.classifier[-1] = nn.Linear(
            in_features, (dim**2) * n_features
        )

    def forward(self, x):
        y = self.vgg_model(x)
        return y.view(self.dim, self.dim, self.n_features)
