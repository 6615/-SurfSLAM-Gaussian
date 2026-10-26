"""Lightweight 3D Gaussian map export utilities.

This module deliberately has no CUDA or Gaussian-rasterizer dependency.  It
turns the metric points already produced by SurfSLAM into the parameter layout
used by the reference 3D Gaussian Splatting implementation.  A renderer such
as ``gsplat`` can consume the resulting PLY/NPZ files later, while the normal
SurfSLAM point-cloud path remains unchanged.

The exported PLY stores the standard fields used by 3DGS:
``xyz``, SH DC/rest coefficients, inverse-sigmoid opacity, log scales, and a
unit quaternion.  With ``sh_degree=0`` the file contains RGB encoded in the
three DC coefficients and no higher-order appearance terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


# The constant used by the original 3DGS code to map RGB values to the
# zeroth-order spherical-harmonic coefficient.
SH_C0 = 0.28209479177387814


def _as_points(points: Any) -> np.ndarray:
    """Return finite ``(N, 3)`` float32 points or raise a useful error."""

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array).all(axis=1)
    return np.ascontiguousarray(array[valid])


def _as_colors(colors: Any, count: int) -> np.ndarray:
    """Normalize colors to finite float32 RGB values in ``[0, 1]``."""

    if colors is None:
        return np.full((count, 3), 0.5, dtype=np.float32)

    array = np.asarray(colors)
    if array.ndim != 2 or array.shape[1] < 3 or array.shape[0] != count:
        raise ValueError(
            f"colors must have shape (N, 3) matching points, got {array.shape} "
            f"for N={count}"
        )
    array = np.asarray(array[:, :3], dtype=np.float32)
    # FrameTracker currently supplies uint8 colors, while callers of this
    # module often use normalized floats.  Accept both representations.
    finite = np.isfinite(array).all(axis=1)
    if not finite.all():
        array = array.copy()
        array[~finite] = 0.5
    if array.size and float(array.max()) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _voxel_reduce(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Average points/colors in a voxel grid without external dependencies."""

    if voxel_size is None or voxel_size <= 0.0 or len(points) == 0:
        return points, colors

    voxel = np.floor(points / float(voxel_size)).astype(np.int64)
    unique, inverse = np.unique(voxel, axis=0, return_inverse=True)
    sums_points = np.zeros((len(unique), 3), dtype=np.float64)
    sums_colors = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(sums_points, inverse, points)
    np.add.at(sums_colors, inverse, colors)
    reduced_points = sums_points / counts[:, None]
    reduced_colors = sums_colors / counts[:, None]
    return reduced_points.astype(np.float32), reduced_colors.astype(np.float32)


def _limit_count(
    points: np.ndarray,
    colors: np.ndarray,
    max_gaussians: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically cap the map while retaining a spatially mixed sample."""

    if max_gaussians is None or max_gaussians <= 0 or len(points) <= max_gaussians:
        return points, colors
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(points), size=int(max_gaussians), replace=False))
    return points[keep], colors[keep]


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-4, 1.0 - 1e-4))
    return float(np.log(value / (1.0 - value)))


def build_gaussians(
    points: Any,
    colors: Any = None,
    *,
    voxel_size: Optional[float] = 0.02,
    max_gaussians: Optional[int] = 200_000,
    opacity: float = 0.8,
    scale: Optional[Union[float, Sequence[float]]] = None,
    sh_degree: int = 0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Create a standard 3DGS parameter dictionary from world points.

    Args:
        points: World-frame point array with shape ``(N, 3)``.
        colors: Optional RGB array, either uint8 ``[0, 255]`` or float ``[0, 1]``.
        voxel_size: Voxel edge length used to merge nearby points.  Set to
            ``None`` or a non-positive value to disable reduction.
        max_gaussians: Maximum number of output Gaussians.  ``None`` or a
            non-positive value disables the cap.
        opacity: Initial visible opacity in ``(0, 1)``.  It is stored as the
            inverse-sigmoid value expected by common 3DGS implementations.
        scale: Initial Gaussian standard deviation in metres.  A scalar gives
            isotropic Gaussians; a length-3 sequence gives per-axis values.
            The default is half the voxel size (or 1 cm when voxel reduction is
            disabled).
        sh_degree: Spherical-harmonic degree.  RGB is stored in DC terms and
            higher-order terms are initialized to zero.
        seed: Seed used when enforcing ``max_gaussians``.

    Returns:
        A dictionary with ``xyz``, ``normals``, ``features_dc``,
        ``features_rest``, ``opacity``, ``scales``, ``rotations``, and metadata
        arrays.  Scale and opacity values use the log/logit parameterization of
        the reference 3DGS implementation.
    """

    if int(sh_degree) != sh_degree or sh_degree < 0:
        raise ValueError(f"sh_degree must be a non-negative integer, got {sh_degree}")
    sh_degree = int(sh_degree)
    points_array = _as_points(points)
    # Colors are filtered along with points.  For invalid point rows, use the
    # corresponding original colors only when the row count is unchanged.
    if colors is None:
        colors_array = _as_colors(None, len(points_array))
    else:
        raw_points = np.asarray(points)
        raw_colors = _as_colors(colors, len(raw_points))
        finite = np.isfinite(np.asarray(raw_points, dtype=np.float32)).all(axis=1)
        colors_array = raw_colors[finite]
        if len(colors_array) != len(points_array):
            raise ValueError("points and colors contain inconsistent finite rows")

    points_array, colors_array = _voxel_reduce(points_array, colors_array, voxel_size)
    points_array, colors_array = _limit_count(
        points_array, colors_array, max_gaussians, seed
    )

    if scale is None:
        scale_value = max(float(voxel_size) * 0.5, 1e-3) if voxel_size and voxel_size > 0 else 1e-2
        scale_array = np.full((len(points_array), 3), scale_value, dtype=np.float32)
    else:
        scale_values = np.asarray(scale, dtype=np.float32)
        if scale_values.ndim == 0:
            scale_values = np.repeat(scale_values.reshape(1), 3)
        if scale_values.shape != (3,):
            raise ValueError(f"scale must be a scalar or length-3 sequence, got {scale_values.shape}")
        if not np.isfinite(scale_values).all() or (scale_values <= 0).any():
            raise ValueError("scale values must be finite and strictly positive")
        scale_array = np.broadcast_to(scale_values, (len(points_array), 3)).copy()

    # PLY/3DGS stores trainable log-scales and inverse-sigmoid opacities.
    scales = np.log(np.maximum(scale_array, 1e-6)).astype(np.float32)
    opacity_array = np.full((len(points_array), 1), _logit(opacity), dtype=np.float32)

    features_dc = ((colors_array - 0.5) / SH_C0).reshape(len(points_array), 3, 1)
    rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    features_rest = np.zeros((len(points_array), rest_count), dtype=np.float32)

    return {
        "xyz": np.ascontiguousarray(points_array.astype(np.float32)),
        "normals": np.zeros((len(points_array), 3), dtype=np.float32),
        "features_dc": np.ascontiguousarray(features_dc.astype(np.float32)),
        "features_rest": features_rest,
        "opacity": opacity_array,
        "scales": scales,
        "rotations": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (len(points_array), 1),
        ),
        # Keeping normalized RGB makes the NPZ useful without decoding SH.
        "colors": np.ascontiguousarray(colors_array.astype(np.float32)),
        "sh_degree": np.asarray(sh_degree, dtype=np.int32),
        "voxel_size": np.asarray(-1.0 if voxel_size is None else voxel_size, dtype=np.float32),
    }


def _validate_gaussians(gaussians: Dict[str, np.ndarray]) -> int:
    required = ("xyz", "normals", "features_dc", "features_rest", "opacity", "scales", "rotations")
    missing = [key for key in required if key not in gaussians]
    if missing:
        raise ValueError(f"Gaussian map missing fields: {', '.join(missing)}")
    count = len(np.asarray(gaussians["xyz"]))
    for key in required:
        value = np.asarray(gaussians[key])
        if len(value) != count:
            raise ValueError(f"Gaussian field {key!r} has {len(value)} rows, expected {count}")
    return count


def save_gaussian_ply(gaussians: Dict[str, np.ndarray], path: Union[str, Path]) -> Path:
    """Write a binary little-endian PLY compatible with common 3DGS loaders."""

    count = _validate_gaussians(gaussians)
    xyz = np.asarray(gaussians["xyz"], dtype=np.float32)
    normals = np.asarray(gaussians["normals"], dtype=np.float32)
    dc = np.asarray(gaussians["features_dc"], dtype=np.float32).reshape(count, 3)
    rest = np.asarray(gaussians["features_rest"], dtype=np.float32).reshape(count, -1)
    opacity = np.asarray(gaussians["opacity"], dtype=np.float32).reshape(count, 1)
    scales = np.asarray(gaussians["scales"], dtype=np.float32).reshape(count, 3)
    rotations = np.asarray(gaussians["rotations"], dtype=np.float32).reshape(count, 4)

    dtype_fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    dtype_fields += [(f"f_dc_{index}", "f4") for index in range(3)]
    dtype_fields += [(f"f_rest_{index}", "f4") for index in range(rest.shape[1])]
    dtype_fields += [("opacity", "f4")]
    dtype_fields += [(f"scale_{index}", "f4") for index in range(3)]
    dtype_fields += [(f"rot_{index}", "f4") for index in range(4)]

    record = np.empty(count, dtype=np.dtype(dtype_fields))
    record["x"], record["y"], record["z"] = xyz.T if count else ([], [], [])
    record["nx"], record["ny"], record["nz"] = normals.T if count else ([], [], [])
    for index in range(3):
        record[f"f_dc_{index}"] = dc[:, index]
    for index in range(rest.shape[1]):
        record[f"f_rest_{index}"] = rest[:, index]
    record["opacity"] = opacity[:, 0]
    for index in range(3):
        record[f"scale_{index}"] = scales[:, index]
    for index in range(4):
        record[f"rot_{index}"] = rotations[:, index]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
    ]
    header.extend(f"property float {name}" for name, _ in dtype_fields)
    header.extend(("end_header", ""))
    with output.open("wb") as handle:
        handle.write(("\n".join(header)).encode("ascii"))
        record.tofile(handle)
    return output


def save_gaussian_npz(gaussians: Dict[str, np.ndarray], path: Union[str, Path]) -> Path:
    """Write the full Gaussian parameter dictionary as a compressed NPZ."""

    _validate_gaussians(gaussians)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **gaussians)
    return output


def save_gaussian_map(
    points: Any,
    colors: Any,
    ply_path: Union[str, Path],
    npz_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Build and save a Gaussian map, returning its in-memory parameters."""

    gaussians = build_gaussians(points, colors, **kwargs)
    save_gaussian_ply(gaussians, ply_path)
    if npz_path is not None:
        save_gaussian_npz(gaussians, npz_path)
    return gaussians


class GaussianMapWriter:
    """Configurable writer used by :class:`FrameTracker`.

    This is intentionally an export-time writer.  It does not add a CUDA
    rasterizer or an optimization loop to the tracking path, so selecting the
    Gaussian representation cannot slow down or alter pose estimation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(config or {})
        self.voxel_size = config.get("voxel_size", 0.03)
        self.max_gaussians = config.get("max_gaussians", 500_000)
        self.opacity = config.get("opacity", 0.8)
        self.scale_multiplier = float(config.get("scale_multiplier", 1.0))
        self.scale = config.get("scale", None)
        self.sh_degree = int(config.get("sh_degree", 0))
        self.seed = int(config.get("seed", 0))

        if self.voxel_size is not None and float(self.voxel_size) < 0:
            raise ValueError("mapper.gaussian.voxel_size must be non-negative or null")
        if self.max_gaussians is not None and int(self.max_gaussians) < 0:
            raise ValueError("mapper.gaussian.max_gaussians must be non-negative or null")
        if not 0.0 < float(self.opacity) < 1.0:
            raise ValueError("mapper.gaussian.opacity must be between 0 and 1")
        if self.scale_multiplier <= 0.0:
            raise ValueError("mapper.gaussian.scale_multiplier must be positive")

    def _scale(self) -> Optional[Union[float, Sequence[float]]]:
        if self.scale is not None:
            values = np.asarray(self.scale, dtype=np.float32) * self.scale_multiplier
            return values.tolist() if values.ndim else float(values)
        # Match build_gaussians' default half-voxel radius while honoring the
        # user-facing multiplier.  Keep a small fallback when voxel reduction
        # is disabled.
        base = (
            0.5 * float(self.voxel_size)
            if self.voxel_size is not None and float(self.voxel_size) > 0.0
            else 1e-2
        )
        return base * self.scale_multiplier

    def write(
        self, ply_path: Union[str, Path], points: Any, colors: Any = None
    ) -> Dict[str, Any]:
        ply_path = Path(ply_path)
        npz_path = ply_path.with_suffix(".npz")
        gaussian_params = save_gaussian_map(
            points,
            colors,
            ply_path,
            npz_path,
            voxel_size=self.voxel_size,
            max_gaussians=self.max_gaussians,
            opacity=self.opacity,
            scale=self._scale(),
            sh_degree=self.sh_degree,
            seed=self.seed,
        )
        return {
            "count": int(len(gaussian_params["xyz"])),
            "ply_path": ply_path,
            "npz_path": npz_path,
            # Online mode uses the reduced representation as its next bounded
            # accumulator instead of retaining every raw depth pixel forever.
            "points": gaussian_params["xyz"],
            "colors": gaussian_params["colors"],
        }


__all__ = [
    "SH_C0",
    "build_gaussians",
    "save_gaussian_ply",
    "save_gaussian_npz",
    "save_gaussian_map",
    "GaussianMapWriter",
]
