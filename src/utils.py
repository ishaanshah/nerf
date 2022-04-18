import torch
from .model import NeRFModel
from torch import Tensor
from typing import Tuple


def positional_encoding(vec: Tensor, L: int) -> Tensor:
    """Postionally encode a batch of numbers

    Inputs -
        vec [B]: Batch of features to perform positional encoding on
        L: Number of terms to use in positional encoding
    Outputs -
        res [B * 2L]: Encoded features
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
        o [h * w * 3]: Origin of all the rays in the image
        d [h * w * 3]: Direction of all the rays in the image

    Note -
        Taken from original NeRF implementation
        https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L123
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="xy")
    d = torch.dstack([(x - w / 2) / f, (y - h / 2) / f, -torch.ones_like(y)])
    d = (d.unsqueeze(2) * mat[:3, :3]).sum(dim=-1)
    o = torch.broadcast_to(mat[:3, 3], (w, h, 3))
    # Swap w,h axis
    o = torch.transpose(o, 0, 1)
    d = torch.transpose(d, 0, 1)
    return o, d


def render(
    o: Tensor,
    d: Tensor,
    t: Tensor,
    model_coarse: NeRFModel,
    model_fine: NeRFModel,
    Lx: int,
    Ld: int,
    white_bck: bool,
) -> Tuple[Tensor, Tensor]:
    """Render color along a ray

    Inputs -
        o [B * 3]: Origin of the ray
        d [B * 3]: Direction of the ray
        t [B * n]: Points to sample on the ray
        model_coarse/fine: Coarse and fine NeRF models
        Lx/Ld: Number of components to use for positional encoding
        white_bck: Whether the image has white background
    """
    B, n = t.shape
    pos = torch.broadcast_to(o, (n, B, 3)).transpose(0, 1)
    pos = pos + t[..., None] * d[:, None, :]  # B * n * 3

    # Perform positional encoding on position and direction vectors
    encoded_pos = torch.zeros(B, n, Lx * 6)
    encoded_dir = torch.zeros(B, Ld * 6)
    for i in range(3):
        encoded_pos[:, :, i * 2 * Lx : (i + 1) * 2 * Lx] = positional_encoding(
            pos[:, :, i].flatten(), Lx
        ).reshape(B, n, -1)
        encoded_dir[:, i * 2 * Ld : (i + 1) * 2 * Ld] = positional_encoding(
            d[:, i].flatten(), Ld
        ).reshape(B, -1)
    encoded_dir = encoded_dir[:, None, :].repeat(1, n, 1)

    # Get color with coarse sampling
    sigma, color = model_coarse(
        encoded_pos.reshape(-1, Lx * 6), encoded_dir.reshape(-1, Ld * 6)
    )
    sigma = sigma.reshape(B, n)
    color = color.reshape(B, n)

    # Apply rendering equation
    delta = t[:, 1:] - t[:, :-1]
    delta = torch.cat([delta, 1e9 * torch.ones_like(delta[:, :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    delta = delta * torch.norm(dir.unsqueeze(1), dim=-1)

    sig_del = sigma * delta
    alpha = 1 - torch.exp(-sig_del)
    t = torch.cumprod(1 - alpha + 1e-10, dim=-1)
    t = torch.cat((torch.ones_like(alpha[:, :1]), t[:, :-1]), dim=-1)

    w = alpha * t
    c = w.unsqueeze(-1) * color

    # If background is white set color to 1 where alpha is 0
    if white_bck:
        c = c + 1 - w.sum(1)

    return c, w
    # TODO: Do fine sampling


def sample_coarse(B: int, n: int, near: Tensor, far: Tensor) -> Tensor:
    """Use stratified sampling to get 'n' samples along the ray

    Inputs -
        B: Batch size
        n: Number of points to sample
        near/far [B]: Bounds of the scene
    Ouptut -
        samples [B * n]: The sampled points
    """
    samples = torch.linspace(0, 1, n+1)
    samples = samples.expand(B, n+1)
    disp = torch.rand_like(samples) / (n+1)
    samples = (samples + disp)[...,:-1]
    samples = near.unsqueeze(-1) * (1 - samples) + far.unsqueeze(-1) * samples
    return samples


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
    # TODO: Check if function has to be differentiable, if not use Categorical
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
