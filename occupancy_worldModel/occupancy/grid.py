from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BEVGrid:
    """Ego-centric BEV in the same frame as world-model `current_local_xy`.

    After rotating world-minus-ego-origin by -ego_yaw:
      x = forward (meters)
      y = left (meters)

    Occupancy tensors are [..., H, W] = [..., y_cells, x_cells].
    """

    x_min: float = -10.0
    x_max: float = 40.0
    y_min: float = -20.0
    y_max: float = 20.0
    cell_size: float = 0.5

    def __post_init__(self) -> None:
        if float(self.cell_size) <= 0:
            raise ValueError(f"cell_size must be > 0, got {self.cell_size}")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("BEV extents must satisfy max > min")

    @property
    def width(self) -> int:
        return int(round((float(self.x_max) - float(self.x_min)) / float(self.cell_size)))

    @property
    def height(self) -> int:
        return int(round((float(self.y_max) - float(self.y_min)) / float(self.cell_size)))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def cell_centers(self, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        xs = self.x_min + (torch.arange(self.width, device=device, dtype=dtype) + 0.5) * self.cell_size
        ys = self.y_min + (torch.arange(self.height, device=device, dtype=dtype) + 0.5) * self.cell_size
        return xs, ys

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "cell_size": float(self.cell_size),
        }

    @classmethod
    def from_config(cls, config: object) -> "BEVGrid":
        return cls(
            x_min=float(getattr(config, "occ_x_min")),
            x_max=float(getattr(config, "occ_x_max")),
            y_min=float(getattr(config, "occ_y_min")),
            y_max=float(getattr(config, "occ_y_max")),
            cell_size=float(getattr(config, "occ_cell_size")),
        )
