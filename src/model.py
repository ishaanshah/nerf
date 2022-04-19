import torch
from typing import Tuple
from torch import nn
from torch import Tensor


class NeRFModel(nn.Module):
    def __init__(self, position_dim: int, direction_dim: int) -> None:
        """
        Inputs -
            position_dim: Number of elements used to encode position
            direction_dim: Number of elements used to encode viewing direcion
        """
        super(NeRFModel, self).__init__()

        pre_skip = [nn.Linear(position_dim, 256), nn.ReLU()]
        for _ in range(3):
            pre_skip.append(nn.Linear(256, 256))
            pre_skip.append(nn.ReLU())
        self.pre_skip = nn.Sequential(*pre_skip)

        self.skip = nn.Sequential(nn.Linear(256 + position_dim, 256), nn.ReLU())

        post_skip = []
        for _ in range(3):
            post_skip.append(nn.Linear(256, 256))
            post_skip.append(nn.ReLU())
        self.post_skip = nn.Sequential(*post_skip)

        self.post_depth = nn.Linear(256, 256)
        self.post_dir = nn.Sequential(nn.Linear(256 + direction_dim, 128), nn.ReLU())

        # The original paper suggests ReLU however it has been observed
        # that Softplus makes the training more stable and reliable
        # for blender dataset which consists a lot of white background
        # Refer: https://github.com/bmild/nerf/issues/29
        self.output_depth = nn.Sequential(
            nn.Linear(in_features=256, out_features=1), nn.Softplus()
        )
        self.output_color = nn.Sequential(
            nn.Linear(in_features=128, out_features=3), nn.Sigmoid()
        )

    def forward(self, x: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inputs -
            x [B * (6*Lx)]: Positionally encoded position
            d [B * (6*Ld)]: Positionally encoded direction vector
        Outputs -
            sigma [B]: Density at the given position
            color [B]: Color at the given postion
        """
        t = self.pre_skip(x)
        x = torch.cat((t, x), dim=1)
        x = self.skip(x)
        x = self.post_skip(x)

        sigma = self.output_depth(x)
        x = self.post_depth(x)
        x = torch.cat((d, x), dim=1)
        x = self.post_dir(x)

        color = self.output_color(x)

        return sigma, color
