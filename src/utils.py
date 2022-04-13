import torch
import numpy as np
from torch import Tensor
from typing import Tuple


def positional_encoding(vec, L=5):
    """Postionally encode a batch of numbers

    Inputs -
        vec [N]: Batch of features to perform positional encoding on
        L: Number of terms to use in positional encoding
    Outputs -
        res [N * L]: Encoded features
    """
    powers = torch.pow(2, torch.arange(L))
    x = torch.pi * torch.unsqueeze(vec, dim=1) * powers

    return torch.concat((torch.sin(x), torch.cos(x)), dim=1)


def get_rays(mat: Tensor, f: float, h: int, w: int) -> Tuple[Tensor, Tensor]:
    """Returns 'o' and 'd' for all pixels in an image

    Inputs -
        mat: World matrix of camera
        f: Focal length of camera
        h, w: Dimensions of image
    Outputs -
        o [w * h * 3]: Origin of all the rays in the image
        d [w * h * 3]: Direction of all the rays in the image

    Note -
        Taken from original NeRF implementation
        https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L123
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    d = torch.dstack([(x - w / 2) / f, (y - h / 2) / f, -torch.ones_like(y)])
    d = (d.unsqueeze(2) * mat[:3, :3]).sum(dim=-1)
    o = torch.broadcast_to(mat[:3, 3], (w, h))
    return o, d


def render():
    """Takes in as input position and direction of ray, and outputs a color

    Inputs -
        postions []:
    """
    pass


def sample_coarse(n: int) -> Tensor:
    """Use stratified sampling to get 'n' samples along the ray

    Inputs -
        n: Number of points to sample
    Ouptut -
        samples: The sampled points
    """
    samples = []
    for i in range(n):
        samples.append(np.random.uniform(i / n, (i + 1) / n))
    return torch.Tensor(samples)


def sample_fine(nf: int, coarse: Tensor, weights: Tensor) -> Tensor:
    """Use inverse transform sampling to sample points
    based upon distribution of density along a ray

    Inputs -
        nf: Number of points to sample excluding the coarse points
        coarse [B * nc]: Coarse samples
        weights [B * nc]: Weights of samples

    Note -
        Taken from original NeRF implementation
        https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L183

    """
    # TODO: Check if function has to be differentiable
    # Prevent NaN
    weights = weights + 1e-5

    # Find CDF
    cdf = torch.cumsum(weights, dim=-1)

    # Normalize to sum up to 1
    cdf = cdf / cdf[:, -1]
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    # Sample uniformly
    u = torch.rand(coarse.shape[0], nf)
    u.contiguous()

    # Find idx where 'u' lies between two values of cdf
    idx = torch.searchsorted(cdf, u, side="right")
    lo = torch.clamp_min(idx - 1, 0)
    hi = torch.clamp_max(idx, coarse.shape[1])

    idx_sampled = torch.stack([lo, hi], dim=-1).reshape(coarse.shape[0], 2 * nf)
    cdf_g = torch.gather(cdf, 1, idx_sampled).view(coarse.shape[0], nf, 2)
    bins_g = torch.gather(coarse, 1, idx_sampled).view(coarse.shape[0], nf, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < 1e-5] = 1

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )

    return samples


def transform_to_ndc(
    o: Tensor, d: Tensor, near: float, f: float, h: float, w: float
) -> Tuple[Tensor, Tensor]:
    """Transforms o and d to NDC space

    Inputs:
        o [B * 3]: Coordinates of origin
        d [B * 3]: Coordinates of direcion
        near: Z coordinate of near plane
        f: focal length of camera
        h: height of the image
        w: width of the image

    Outputs:
        o' [B * 3]: Coordinates of origin in NDC
        d' [B * 3]: Coordinates of direction in NDC
    """
    op = torch.zeros_like(o)
    dp = torch.zeros_like(d)

    # Shift 'o' to '-n'
    no = o + (-(near + o[:, 2]) / d[:, 2]) * d

    # Calculate o'
    op[:, 0] = -f / (w / 2) * (no[:, 0] / no[:, 2])
    op[:, 1] = -f / (h / 2) * (no[:, 1] / no[:, 2])
    op[:, 2] = 1 + (2 * near / no[:, 2])

    # Calculate d'
    dp[:, 0] = -f / (w / 2) * ((d[:, 0] / d[:, 2]) - (no[:, 0] / no[:, 2]))
    dp[:, 1] = -f / (h / 2) * ((d[:, 1] / d[:, 2]) - (no[:, 1] / no[:, 2]))
    dp[:, 2] = -2 * near / no[:, 2]

    return op, dp
