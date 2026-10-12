"""OpenCLIP 输入适配与数值审计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


@dataclass(frozen=True)
class TensorStats:
    shape: str
    minimum: float
    maximum: float
    mean: float
    std: float
    has_nan: bool
    has_inf: bool


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    """生成可直接写入 CSV 的输入 tensor 统计。"""
    tensor = value.detach().float().cpu()
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        minimum = maximum = mean = std = float("nan")
    else:
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        mean = float(finite.mean().item())
        std = float(finite.std(unbiased=False).item())
    return asdict(
        TensorStats(
            shape="x".join(str(part) for part in tensor.shape),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            has_nan=bool(torch.isnan(tensor).any().item()),
            has_inf=bool(torch.isinf(tensor).any().item()),
        )
    )


class CLIPInputAdapter:
    """在旧输入和 OpenCLIP 官方验证预处理之间显式切换。"""

    MODES = {"legacy", "official"}

    def __init__(self, mode: str, official_preprocess=None):
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"未知 CLIP preprocess mode: {mode}")
        if mode == "official" and official_preprocess is None:
            raise ValueError("official 模式必须提供 open_clip 返回的验证预处理")
        self.mode = mode
        self.official_preprocess = official_preprocess
        self.last_stats: dict[str, Any] | None = None
        self.stats_history: list[dict[str, Any]] = []

    @staticmethod
    def _as_pil(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(
                    f"单张图像 tensor 应为 CHW，实际为 {tuple(tensor.shape)}"
                )
            return to_pil_image(tensor).convert("RGB")
        array = np.asarray(image)
        if (
            array.ndim == 3
            and array.shape[0] in {1, 3, 4}
            and array.shape[-1] not in {1, 3, 4}
        ):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0)
            array = np.rint(array * 255.0).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def __call__(self, image: Any) -> torch.Tensor:
        if self.mode == "legacy":
            output = (
                image
                if isinstance(image, torch.Tensor)
                else torch.as_tensor(np.asarray(image))
            )
            output = output.float()
        elif isinstance(image, torch.Tensor) and image.ndim == 4:
            output = torch.stack(
                [self.official_preprocess(self._as_pil(item)) for item in image], dim=0
            )
        else:
            output = self.official_preprocess(self._as_pil(image))
        self.last_stats = tensor_stats(output)
        self.stats_history.append(self.last_stats)
        return output
