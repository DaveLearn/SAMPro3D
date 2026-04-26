"""Debug visualization utilities for the SAMPro3D external segmenter.

Gated by the SAMPRO3D_DEBUG environment variable. When SAMPRO3D_DEBUG=1,
three checkpoints save artifacts (images, PLY meshes) to a debug output
directory. When disabled, all methods are no-ops with zero overhead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("sampro3d-segmenter")


def is_debug_enabled() -> bool:
    return os.environ.get("SAMPRO3D_DEBUG", "0") == "1"


def _random_colors_for_labels(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    labels_int = labels.astype(np.int64)
    colors = np.empty((labels_int.shape[0], 3), dtype=np.float64)

    bg_mask = labels_int <= 0
    colors[bg_mask] = np.array([0.9, 0.9, 0.9])

    fg = labels_int[~bg_mask]
    if fg.size:
        hashed = (fg * 2654435761 + seed) & 0xFFFFFFFF
        r = ((hashed >> 16) & 0xFF) / 255.0
        g = ((hashed >> 8) & 0xFF) / 255.0
        b = (hashed & 0xFF) / 255.0
        colors[~bg_mask] = np.stack([r, g, b], axis=-1)

    return colors


class DebugVisualizer:
    """Saves debug artifacts for three pipeline stages when SAMPRO3D_DEBUG=1."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.enabled = is_debug_enabled()
        self.output_dir = output_dir
        if self.enabled:
            if output_dir is None:
                output_dir = Path("/tmp/sampro3d_debug")
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug visualizer ENABLED -- output dir: %s", self.output_dir)
        else:
            logger.debug("Debug visualizer disabled (set SAMPRO3D_DEBUG=1 to enable)")

    # ------------------------------------------------------------------
    # Checkpoint 1: Verify temp ScanNet export
    # ------------------------------------------------------------------
    def save_scannet_dataset(self, dataset_root: Path, scene_id: str) -> None:
        if not self.enabled or self.output_dir is None:
            return
        import imageio.v2 as imageio

        scene_dir = dataset_root / scene_id
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        pose_dir = scene_dir / "pose"

        n_color = len(list(color_dir.glob("*.jpg"))) if color_dir.exists() else 0
        n_depth = len(list(depth_dir.glob("*.png"))) if depth_dir.exists() else 0
        n_pose = len(list(pose_dir.glob("*.txt"))) if pose_dir.exists() else 0
        ply_exists = (scene_dir / f"{scene_id}_vh_clean_2.ply").exists()
        intrinsics_path = dataset_root / "intrinsics.txt"

        logger.info(
            "[DEBUG] ScanNet dataset: %d color, %d depth, %d pose, PLY=%s, intrinsics=%s",
            n_color,
            n_depth,
            n_pose,
            ply_exists,
            intrinsics_path.exists(),
        )

        # Save first frame as verification
        out = self.output_dir / "scannet_export"
        out.mkdir(exist_ok=True)
        first_color = sorted(color_dir.glob("*.jpg"))[0] if n_color else None
        first_depth = sorted(depth_dir.glob("*.png"))[0] if n_depth else None
        if first_color is not None:
            imageio.imwrite(out / "first_color.png", imageio.imread(first_color))
        if first_depth is not None:
            depth = imageio.imread(first_depth)
            depth_vis = (depth / 16.0).clip(0, 255).astype(np.uint8)  # rough scale to uint8
            imageio.imwrite(out / "first_depth.png", depth_vis)

    # ------------------------------------------------------------------
    # Checkpoint 2: Segmented mesh (raw and filtered)
    # ------------------------------------------------------------------
    def save_segmented_mesh(self, mesh, vertex_labels: np.ndarray, filename: str = "mesh_segmented.ply") -> None:
        if not self.enabled or self.output_dir is None:
            return
        import copy

        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(vertex_labels, seed=77)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        n_labels = len(np.unique(vertex_labels[vertex_labels > 0]))
        logger.info("[DEBUG] Segmented mesh: %d instances -> %s", n_labels, path)

    # ------------------------------------------------------------------
    # Checkpoint 3: Back-projected pixel masks
    # ------------------------------------------------------------------
    def save_pixel_masks(self, frames, instance_groups: Dict[str, np.ndarray], max_frames: int = 6) -> None:
        if not self.enabled or self.output_dir is None:
            return
        import imageio.v2 as imageio

        out = self.output_dir / "pixel_masks"
        out.mkdir(exist_ok=True)

        saved = 0
        for frame in frames:
            if saved >= max_frames:
                break
            mask = instance_groups.get(frame.name)
            if mask is None:
                continue

            colors = _random_colors_for_labels(mask.ravel(), seed=42).reshape(mask.shape + (3,))
            colors_uint8 = (colors * 255).astype(np.uint8)
            rgb = frame.color.cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            imageio.imwrite(out / f"{frame.name}_rgb.png", rgb_uint8)
            imageio.imwrite(out / f"{frame.name}_pixel_masks.png", colors_uint8)
            saved += 1

        logger.info("[DEBUG] Saved %d back-projected pixel mask visualizations -> %s", saved, out)
