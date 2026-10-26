"""Optional differentiable Gaussian refinement backed by gsplat.

The regular Gaussian writer is dependency-free and only initializes/export
primitives.  This module adds the photometric part: gsplat rasterizes the
current Gaussian set into the latest RGB/depth keyframe and PyTorch optimizes
the Gaussian colors and (optionally) means through the rendered loss.

It is intentionally a small local refiner rather than a second pose graph.
SurfSLAM remains responsible for metric depth, tracking, and loop closure;
the provisional Gaussian pose can be replaced by the final optimized pose at
shutdown.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _logit(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(values) - torch.log1p(-values)


class GsplatPhotometricRefiner:
    """Run a bounded differentiable RGB/depth refinement step."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda:0"):
        self.config = dict(config or {})
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.iterations = max(1, int(self.config.get("optimization_iterations", 3)))
        self.learning_rate = float(self.config.get("optimization_lr", 0.01))
        self.mean_learning_rate = float(
            self.config.get("mean_learning_rate", self.learning_rate * 0.1)
        )
        self.optimize_means = bool(self.config.get("optimize_means", True))
        self.rgb_weight = float(self.config.get("rgb_loss_weight", 1.0))
        self.depth_weight = float(self.config.get("depth_loss_weight", 0.1))
        self.max_optimize_gaussians = int(
            self.config.get("max_optimize_gaussians", 50_000)
        )
        self.render_scale = float(self.config.get("render_scale", 0.5))
        self._gsplat = None
        try:
            from gsplat import rasterization

            self._gsplat = rasterization
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._gsplat is not None and self.device.type == "cuda"

    def _sample(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.max_optimize_gaussians <= 0 or len(points) <= self.max_optimize_gaussians:
            return points, colors
        # The writer already performs voxel reduction.  A deterministic stride
        # keeps this optional optimization bounded for large live maps.
        indices = np.linspace(
            0, len(points) - 1, self.max_optimize_gaussians, dtype=np.int64
        )
        return points[indices], colors[indices]

    def refine(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        world_T_cam: np.ndarray,
        intrinsics: np.ndarray,
        target_rgb: torch.Tensor,
        target_depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Optimize colors and optionally means against one current keyframe."""
        if not self.available:
            raise RuntimeError(
                "Gaussian mode 'gsplat' requires gsplat and a CUDA device. "
                "Install requirements-gaussian.txt in the SurfSLAM environment."
            )
        points, colors = self._sample(
            np.asarray(points, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
        )
        if len(points) == 0:
            return {"points": points, "colors": colors, "loss": None, "iterations": 0}

        device = self.device
        means = torch.tensor(points, device=device, dtype=torch.float32)
        means.requires_grad_(self.optimize_means)
        color_logits = _logit(torch.tensor(colors, device=device, dtype=torch.float32))
        color_logits.requires_grad_(True)

        voxel_size = float(self.config.get("voxel_size", 0.03) or 0.03)
        gaussian_scale = max(voxel_size * 0.5, 1e-3)
        scales = torch.full(
            (len(points), 3), gaussian_scale, device=device, dtype=torch.float32
        )
        quats = torch.zeros((len(points), 4), device=device, dtype=torch.float32)
        quats[:, 0] = 1.0
        opacities = torch.full(
            (len(points),), float(self.config.get("opacity", 0.8)),
            device=device,
            dtype=torch.float32,
        )

        rgb = target_rgb.detach().to(device=device, dtype=torch.float32)
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
            rgb = rgb.permute(1, 2, 0)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        depth = None
        if target_depth is not None:
            depth = target_depth.detach().to(device=device, dtype=torch.float32)
            while depth.ndim > 2:
                depth = depth.squeeze(0)

        render_scale = min(max(self.render_scale, 0.1), 1.0)
        height, width = int(rgb.shape[0] * render_scale), int(rgb.shape[1] * render_scale)
        height, width = max(height, 1), max(width, 1)
        if (height, width) != tuple(rgb.shape[:2]):
            rgb = F.interpolate(
                rgb.permute(2, 0, 1)[None], size=(height, width), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)
            if depth is not None:
                depth = F.interpolate(depth[None, None], size=(height, width), mode="nearest")[0, 0]

        K = torch.tensor(intrinsics, device=device, dtype=torch.float32).clone()
        K[:2] *= render_scale
        K = K[None]
        T = torch.tensor(world_T_cam, device=device, dtype=torch.float32)
        viewmats = torch.linalg.inv(T)[None]

        optim_params = [{"params": [color_logits], "lr": self.learning_rate}]
        if self.optimize_means:
            optim_params.append({"params": [means], "lr": self.mean_learning_rate})
        optimizer = torch.optim.Adam(optim_params)
        last_loss = None

        for _ in range(self.iterations):
            optimizer.zero_grad(set_to_none=True)
            rendered, alpha, _ = self._gsplat(
                means,
                quats,
                scales,
                opacities,
                color_logits.sigmoid(),
                viewmats,
                K,
                width,
                height,
                render_mode="RGB+ED" if depth is not None else "RGB",
                sh_degree=None,
                packed=True,
            )
            rendered = rendered[0]
            if rendered.shape[-1] > 3:
                rendered_rgb, rendered_depth = rendered[..., :3], rendered[..., 3]
            else:
                rendered_rgb, rendered_depth = rendered, None

            # Charbonnier-like RGB loss is less brittle under underwater
            # backscatter than a pure squared error.
            rgb_loss = torch.sqrt((rendered_rgb - rgb).square() + 1e-4).mean()
            loss = self.rgb_weight * rgb_loss
            if depth is not None and rendered_depth is not None:
                valid = torch.isfinite(depth) & (depth > 0)
                if valid.any():
                    depth_loss = torch.abs(rendered_depth[valid] - depth[valid]).mean()
                    loss = loss + self.depth_weight * depth_loss
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([means, color_logits], 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        return {
            "points": means.detach().cpu().numpy().astype(np.float32),
            "colors": color_logits.detach().sigmoid().cpu().numpy().astype(np.float32),
            "loss": last_loss,
            "iterations": self.iterations,
        }


__all__ = ["GsplatPhotometricRefiner"]
