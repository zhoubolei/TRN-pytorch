import torch


class FlipChannels:
    """Converts a :math:`(C, T, H, W)` tensor from BGR to RGB or vica versa"""

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.flip(frames, (0,))
