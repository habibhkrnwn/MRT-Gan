"""Mask-guided native-resolution patches; no resized or black-masked D inputs."""
import torch
import torch.nn.functional as F


def patch_indices(mask, size=64, stride=16, count=4, coverage=0.95, random=True, generator=None):
    if size < 32 or stride < 1 or count < 1 or not 0 < coverage <= 1:
        raise ValueError('invalid texture patch parameters')
    if mask.ndim != 4 or mask.shape[1] != 1 or min(mask.shape[-2:]) < size:
        raise ValueError('mask must be N x 1 x H x W with H,W >= patch size')
    occupancy = F.avg_pool2d(mask.float(), size, stride).flatten(1)
    selected = []
    for values in occupancy:
        eligible = torch.where(values >= coverage)[0]
        if not len(eligible):
            raise ValueError(f'no interior texture patch: size={size}, coverage={coverage}; inspect mask')
        if random:
            eligible = eligible[torch.randperm(len(eligible), device=eligible.device, generator=generator)]
        # Every image supplies equal weight, even if it has fewer eligible patches.
        selected.append(eligible.repeat((count + len(eligible) - 1) // len(eligible))[:count])
    return torch.stack(selected)


def extract_patches(image, indices, size=64, stride=16, gray=True):
    # Tensor.unfold is a view; gather only the selected patches instead of
    # materialising all RGB windows on the GPU.
    windows = image.unfold(2, size, stride).unfold(3, size, stride)
    cols = windows.shape[3]
    out = torch.stack([windows[n, :, int(i) // cols, int(i) % cols]
                       for n, row in enumerate(indices) for i in row])
    if gray:
        out = (out * out.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    return out
