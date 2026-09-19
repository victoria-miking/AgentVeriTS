from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EncodedBatch:
    large_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    class_tokens: torch.Tensor
    large_mask: torch.Tensor
    mid_mask: torch.Tensor


def _make_mask(image_size: int, patch_size: int, kernel_size: int) -> torch.Tensor:
    side = image_size // patch_size
    board = torch.arange(side * side, dtype=torch.float32).reshape(1, 1, side, side)
    return F.unfold(board, kernel_size=kernel_size // patch_size, stride=1).squeeze(0).long()


def _pool_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pooled = []
    for i in range(mask.shape[1]):
        ids = mask[:, i].to(tokens.device)
        pooled.append(tokens.index_select(1, ids).mean(dim=1, keepdim=True))
    return torch.cat(pooled, dim=1)


class VisualEncoder:
    """OpenCLIP visual backbone used by the coarse visual screening stage.

    The public inference path deliberately uses the legacy [0,1] rendered tensor
    directly, matching the frozen screening configuration used for REVA.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        image_size: int = 224,
        patch_size: int = 16,
        device: str = "auto",
    ) -> None:
        import open_clip

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            vision_cfg={"output_tokens": True},
        )
        self.model = model.to(self.device).eval()
        self.large_mask = _make_mask(self.image_size, self.patch_size, 48)
        self.mid_mask = _make_mask(self.image_size, self.patch_size, 32)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> EncodedBatch:
        images = images.to(self.device, non_blocking=True).float()
        _, patch_tokens = self.model.encode_image(images)
        class_tokens = patch_tokens.mean(dim=1)
        large = _pool_tokens(patch_tokens, self.large_mask)
        mid = _pool_tokens(patch_tokens, self.mid_mask)
        return EncodedBatch(
            large_tokens=large,
            mid_tokens=mid,
            patch_tokens=patch_tokens,
            class_tokens=class_tokens,
            large_mask=self.large_mask.to(self.device),
            mid_mask=self.mid_mask.to(self.device),
        )
