"""
Modified connected components analysis from ICI loss project
Repository URL: https://github.com/BrainImageAnalysis/ICI-loss
"""

import torch
import torch.nn.functional as F


def connected_components_with_gradients(
        image: torch.Tensor,
        num_iterations: int = 75,
        threshold: float = 0.5,
        spatial_dims: int = 3
) -> torch.Tensor:
    if spatial_dims == 3:
        H, W, Z = image.shape[-3:]

        ## precompute a mask with the valid values
        mask = torch.zeros((1, 1, H, W, Z), dtype=torch.bool, device=image.device)
        mask[image >= threshold] = True
        image = torch.ceil(image * mask)

        B = image.shape[0]
        out = torch.arange(B * H * W * Z, device=image.device, dtype=image.dtype).view(-1, 1, H, W, Z)
        out[~mask] = 0.

        for _ in range(num_iterations):
            out[mask] = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)[mask]

    elif spatial_dims == 2:
        H, W = image.shape[-2:]

        ## precompute a mask with the valid values
        mask = torch.zeros((1, 1, H, W), dtype=torch.bool, device=image.device)
        mask[image >= threshold] = True
        image = torch.ceil(image * mask)

        B = image.shape[0]
        out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
        out[~mask] = 0.

        for _ in range(num_iterations):
            out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return image * out