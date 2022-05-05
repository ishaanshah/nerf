import torch
from .model import NeRFModel
from torch import Tensor
from typing import Optional, Tuple


def positional_encoding(vec: Tensor, L: int) -> Tensor:
    """Postionally encode a batch of numbers

    Inputs -
        vec [B * 3]: Batch of features to perform positional encoding on
        L: Number of terms to use in positional encoding
    Outputs -
        res [B * 6L]: Encoded features

    Note -
        This is different than the original implementation.
        Refer to https://github.com/bmild/nerf/issues/12
    """
    B, _ = vec.shape
    powers = torch.pow(2, torch.arange(L, device=vec.device)) # L
    x = torch.unsqueeze(vec, dim=-1) * powers  # B * 3 * L
    x = x.reshape(B, -1)  # B * 3L
    return torch.concat((torch.sin(x), torch.cos(x)), dim=1)


def get_rays(mat: Tensor, dirs: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns 'o' and 'd' for all pixels in an image

    Inputs -
        mat [4 * 4]: World matrix of camera
        dirs [h * w * 3]: Directions for each pixel in camera coordinates
    Outputs -
        o [h * w * 3]: Origin of all the rays in the image
        d [h * w * 3]: Direction of all the rays in the image
    """
    h, w = dirs.shape[:2]
    d = dirs @ mat[:3, :3].T
    d = d / torch.norm(d, dim=-1, keepdim=True)
    o = torch.broadcast_to(mat[:3, 3], (h, w, 3))
    return o, d


def render(
    o: Tensor,
    d: Tensor,
    t: Tensor,
    model: NeRFModel,
    lx: int,
    ld: int,
    chunk_size: int,
    sdf: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Render color along a ray

    Inputs -
        o [B * 3]: Origin of the ray
        d [B * 3]: Direction of the ray
        t [B * n]: Points to sample on the ray
        model_coarse/fine: Coarse and fine NeRF models
        lx/ld: Number of components to use for positional encoding
        chunk_size: Number of points to process at a time
    Outputs -
        c [B * 3]: Color of ray
        w [B * n]: Weights of points sampled along the ray
    """
    B, n = t.shape
    pos = o[:,None,:].repeat(1, n, 1)
    pos = pos + t[..., None] * d[:, None, :]  # B * n * 3
    dir = d[:, None, :].repeat(1, n, 1)

    #TODO - positionally encode SDF
    # Perform positional encoding on position and direction vectors
    pos = pos.reshape(-1, 3).float()
    dir = dir.reshape(-1, 3).float()

    sigma = torch.zeros(B * n, dtype=torch.float32, device=o.device)
    color = torch.zeros((B * n, 3), dtype=torch.float32, device=o.device)
    for i in range(0, B * n, chunk_size):
        encoded_pos = positional_encoding(pos[i : i + chunk_size], lx).float()
        encoded_dir = positional_encoding(dir[i : i + chunk_size], ld).float()
        # Get color with coarse sampling
        sigma_, color_ = model(
            encoded_pos, encoded_dir
        )
        sigma[i : i + chunk_size] = sigma_.reshape(chunk_size)
        color[i : i + chunk_size] = color_.reshape(chunk_size, 3)

    sigma = sigma.reshape(B, n)
    color = color.reshape(B, n, 3)

    # Apply rendering equation
    delta = t[:, 1:] - t[:, :-1]
    delta = torch.cat([delta, 1e9 * torch.ones_like(delta[:, :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    delta = delta * torch.norm(d.unsqueeze(1), dim=-1)

    sig_del = sigma * delta
    alpha = 1 - torch.exp(-sig_del)
    t = torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10), dim=-1)
    t = torch.cumprod(t, dim=-1)[:,:-1]

    w = alpha * t # B * n
    c = torch.sum(w.unsqueeze(-1) * color, 1)

    return c, w


def sample_coarse(B: int, n: int, near: Tensor, far: Tensor) -> Tensor:
    """Use stratified sampling to get 'n' samples along the ray

    Inputs -
        B: Batch size
        n: Number of points to sample
        near/far [B]: Bounds of the scene
    Ouptut -
        samples [B * n]: The sampled points
    """
    samples = torch.linspace(0, 1, n + 1, device=near.device)
    samples = samples.expand(B, n + 1)
    disp = torch.rand_like(samples) / n
    samples = (samples + disp)[..., :-1]
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
    cdf = cdf / cdf[:, -1:]
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    # Sample uniformly
    u = torch.rand(coarse.shape[0], nf, device=coarse.device)
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
