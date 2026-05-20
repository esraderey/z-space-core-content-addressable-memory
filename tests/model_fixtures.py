import torch
from torch import nn


class TwoMillionParamMLP(nn.Module):
    """Small dense model with 2,002,058 trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 1536),
            nn.GELU(),
            nn.Linear(1536, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_2m_model(seed: int = 2026) -> TwoMillionParamMLP:
    torch.manual_seed(seed)
    return TwoMillionParamMLP().eval()


def build_sparse_weight_patch(
    tensor: torch.Tensor,
    updates: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim != 2:
        raise ValueError("Sparse weight patch expects a 2D tensor")
    flat_positions = torch.linspace(0, tensor.numel() - 1, steps=updates, dtype=torch.int64)
    rows = torch.div(flat_positions, tensor.shape[1], rounding_mode="floor")
    cols = flat_positions.remainder(tensor.shape[1])
    indices = torch.stack((rows, cols), dim=1)
    offsets = torch.linspace(-0.005, 0.005, steps=updates, dtype=tensor.dtype)
    values = tensor[tuple(indices.T)] + offsets
    return indices, values
