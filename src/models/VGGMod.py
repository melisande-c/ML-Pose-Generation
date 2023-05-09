from collections import OrderedDict
from torch import nn


class VGGCoordMod(nn.Module):

    """
    Modifies the final output layer of vgg so that the number of nodes is
    2 * the number of keypoints. This turns the the domain into a regression
    problem.
    """

    def __init__(self, vgg_model, coords_n: int) -> None:
        """
        Parameters
        ----------
        vgg_model: VGG
            Can be pretrained or not.
        coords_n: int
            Number of keepoint coordinates to output
        """
        super().__init__()
        self.coords_n = coords_n
        self.vgg_model = vgg_model
        # get the features from the -4th layer because ReLu and dropout layers
        # come after.
        in_features = self.vgg_model.classifier[-4].out_features
        # replace final layer
        self.vgg_model.classifier[-1] = nn.Linear(in_features, 2 * coords_n)
        self.reshape = nn.Unflatten(1, (self.coords_n, 2))

    def forward(self, x):
        y = self.vgg_model(x)
        # reshape the output so that each row is a key point and the columns
        # are x and y coordinates.
        return self.reshape(y)


class VGGSpatialMod(nn.Module):

    """
    Modifies the last layer so that the output is image shaped with a channel
    for each key point.
    """

    def __init__(self, vgg_model, output_dim: int, n_features: int):
        """
        Parameters
        ----------
        vgg_model: VGG
        output_dims: int
            The (square) output dimensions of the image like output.
        n_features: int
            The number of keypoints, will be the number of channels output.
        """

        super(VGGSpatialMod, self).__init__()
        self.dim = output_dim
        self.n_features = n_features
        self.vgg_model = vgg_model
        # get the features from the -4th layer because ReLu and dropout layers
        # come after.
        in_features = self.vgg_model.classifier[-4].out_features
        self.vgg_model.classifier[-1] = nn.Sequential(
            OrderedDict(
                [
                    # Don't have fully connected as full size because that
                    # takes up a lot of memory
                    ("fc", nn.Linear(in_features, (64**2) * n_features)),
                    ("reshape", nn.Unflatten(1, (n_features, 64, 64))),
                    # upsample to full size
                    ("upsample", nn.Upsample((output_dim, output_dim))),
                ]
            )
        )

    def forward(self, x):
        y = self.vgg_model(x)
        return y
