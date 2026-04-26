"""Standalone debugger for SAMPro3D DEG integration outputs.

This script inspects a SAMPro3D output directory after a run and writes a
diagnostic report plus visualization artifacts that help debug the DEG ->
fake-ScanNet conversion and the downstream SAMPro3D stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, Sequence
import logging
import math

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import tyro

from helpers.debug_visualize import _random_colors_for_labels
from initializerdefs import Observations, SceneSetup
from main import prompt_filter
from psdframe import Frame
from segmenter import (
    _filter_labels_by_workspace,
    _load_sampro3d_labels,
    _render_instance_id_masks,
    _sanitize_scene_id,
    determine_table_instance_id,
    get_workspace_voxels,
)
from utils.main_utils import load_ply, transform_pt_depth_scannet_torch
from utils.sam_utils import MaskData, batched_mask_to_box, calculate_stability_score


logger = logging.getLogger("sampro3d-visualize")


@dataclass
class Args:
    output_dir: tyro.conf.Positional[Path]
    """SAMPro3D run output directory created by segment.py."""

    observations_path: Optional[Path] = None
    """Optional DEG Observations pickle for export comparison."""

    scene_path: Optional[Path] = None
    """Optional DEG SceneSetup pickle for workspace/table-aware diagnostics."""

    save_dir: Optional[Path] = None
    """Optional directory to write visualization artifacts to."""

    stage: Literal["all", "inventory", "export", "prompts", "predictions"] = "all"
    """Restrict work to one stage, or run everything that is available."""

    max_frames: int = 6
    """Maximum number of frame overlays to save per stage."""

    open3d: bool = False
    """Open interactive Open3D viewers for mesh/prompt artifacts when possible."""

    device: str = "cuda"
    """Torch device for prompt reprojection and filter diagnostics."""

    pred_iou_thres: float = 0.7
    """Prompt filter IoU threshold used for stage-1 diagnostics."""

    stability_score_thres: float = 0.6
    """Prompt filter stability threshold used for stage-1 diagnostics."""

    box_nms_thres: float = 0.8
    """Prompt filter NMS threshold used for stage-1 diagnostics."""

    keep_thres: float = 0.4
    """Prompt keep threshold used for stage-1 diagnostics."""

    mask_threshold: float = 0.0
    """SAM mask threshold used for stability and binary mask overlays."""

    post_floor: Optional[bool] = None
    """Override whether to load *_seg_floor.npy; defaults to auto-detect."""


@dataclass
class RunPaths:
    output_dir: Path
    work_root: Path
    dataset_root: Path
    scene_id: Optional[str]
    scene_dir: Optional[Path]
    color_dir: Optional[Path]
    depth_dir: Optional[Path]
    pose_dir: Optional[Path]
    intrinsics_path: Path
    mesh_path: Optional[Path]
    prompt_ply_path: Optional[Path]
    sam_output_scene_dir: Optional[Path]
    pred_dir: Path
    vis_dir: Path
    debug_dir: Path


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


def _sorted_numbered_files(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (_numeric_stem(path), path.name))


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    if limit <= 0 or count <= limit:
        return list(range(count))
    indices = np.linspace(0, count - 1, num=limit, dtype=int)
    return sorted(set(int(idx) for idx in indices))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _scene_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted([path for path in dataset_root.iterdir() if path.is_dir()])


def _resolve_run_paths(output_dir: Path) -> RunPaths:
    work_root = output_dir / "sampro3d_work"
    dataset_root = work_root / "dataset" / "scannet"
    scene_dirs = _scene_dirs(dataset_root)
    scene_dir = scene_dirs[0] if len(scene_dirs) == 1 else None
    scene_id = scene_dir.name if scene_dir is not None else None

    prompt_candidates = sorted((work_root / "init_prompt").glob("*.ply")) if (work_root / "init_prompt").exists() else []
    if scene_id is None and len(prompt_candidates) == 1:
        scene_id = prompt_candidates[0].stem
        scene_dir = dataset_root / scene_id if (dataset_root / scene_id).exists() else None

    sam_output_root = work_root / "sam_output"
    sam_output_scene_dir = sam_output_root / scene_id if scene_id is not None and (sam_output_root / scene_id).exists() else None
    if scene_id is None and sam_output_root.exists():
        child_dirs = sorted([path for path in sam_output_root.iterdir() if path.is_dir()])
        if len(child_dirs) == 1:
            sam_output_scene_dir = child_dirs[0]
            scene_id = child_dirs[0].name
            scene_dir = dataset_root / scene_id if (dataset_root / scene_id).exists() else scene_dir

    color_dir = scene_dir / "color" if scene_dir is not None else None
    depth_dir = scene_dir / "depth" if scene_dir is not None else None
    pose_dir = scene_dir / "pose" if scene_dir is not None else None
    mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply" if scene_dir is not None and scene_id is not None else None
    prompt_ply_path = work_root / "init_prompt" / f"{scene_id}.ply" if scene_id is not None else None
    if prompt_ply_path is not None and not prompt_ply_path.exists():
        prompt_ply_path = None

    return RunPaths(
        output_dir=output_dir,
        work_root=work_root,
        dataset_root=dataset_root,
        scene_id=scene_id,
        scene_dir=scene_dir,
        color_dir=color_dir,
        depth_dir=depth_dir,
        pose_dir=pose_dir,
        intrinsics_path=dataset_root / "intrinsics.txt",
        mesh_path=mesh_path if mesh_path is not None and mesh_path.exists() else None,
        prompt_ply_path=prompt_ply_path,
        sam_output_scene_dir=sam_output_scene_dir,
        pred_dir=work_root / "final_pred",
        vis_dir=work_root / "output_vis",
        debug_dir=output_dir / "debug",
    )


def _inventory(paths: RunPaths) -> tuple[list[str], dict[str, object]]:
    artifacts: dict[str, object] = {
        "work_root": paths.work_root.exists(),
        "dataset_root": paths.dataset_root.exists(),
        "intrinsics": paths.intrinsics_path.exists(),
        "scene_id": paths.scene_id,
        "scene_dir": paths.scene_dir.exists() if paths.scene_dir is not None else False,
        "color_frames": len(list(paths.color_dir.glob("*.jpg"))) if paths.color_dir is not None and paths.color_dir.exists() else 0,
        "depth_frames": len(list(paths.depth_dir.glob("*.png"))) if paths.depth_dir is not None and paths.depth_dir.exists() else 0,
        "pose_frames": len(list(paths.pose_dir.glob("*.txt"))) if paths.pose_dir is not None and paths.pose_dir.exists() else 0,
        "mesh": paths.mesh_path.exists() if paths.mesh_path is not None else False,
        "prompt_ply": paths.prompt_ply_path.exists() if paths.prompt_ply_path is not None else False,
        "sam_output": paths.sam_output_scene_dir.exists() if paths.sam_output_scene_dir is not None else False,
        "final_pred": paths.pred_dir.exists(),
        "output_vis": paths.vis_dir.exists(),
        "debug_dir": paths.debug_dir.exists(),
    }

    lines = [
        "## Inventory",
        f"- `work_root`: {artifacts['work_root']}",
        f"- `dataset_root`: {artifacts['dataset_root']}",
        f"- `intrinsics.txt`: {artifacts['intrinsics']}",
        f"- `scene_id`: {artifacts['scene_id']}",
        f"- `color/*.jpg`: {artifacts['color_frames']}",
        f"- `depth/*.png`: {artifacts['depth_frames']}",
        f"- `pose/*.txt`: {artifacts['pose_frames']}",
        f"- `mesh ply`: {artifacts['mesh']}",
        f"- `init_prompt ply`: {artifacts['prompt_ply']}",
        f"- `sam_output/<scene>`: {artifacts['sam_output']}",
        f"- `final_pred`: {artifacts['final_pred']}",
        f"- `output_vis`: {artifacts['output_vis']}",
        f"- `debug`: {artifacts['debug_dir']}",
    ]
    return lines, artifacts


def _load_intrinsics(path: Path) -> np.ndarray:
    intrinsics = np.loadtxt(path)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (4, 4):
        raise ValueError(f"Expected 4x4 intrinsics matrix at {path}, got {intrinsics.shape}")
    return intrinsics


def _load_pose_files(pose_dir: Optional[Path]) -> list[Path]:
    if pose_dir is None or not pose_dir.exists():
        return []
    return _sorted_numbered_files(list(pose_dir.glob("*.txt")))


def _rotation_stats(rotation: np.ndarray) -> tuple[float, float]:
    identity_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    return identity_error, determinant


def _plot_camera_path(poses: Sequence[np.ndarray], save_path: Path) -> None:
    if not poses:
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    centers = np.stack([pose[:3, 3] for pose in poses], axis=0)
    forward = np.stack([pose[:3, 2] for pose in poses], axis=0)
    up = np.stack([pose[:3, 1] for pose in poses], axis=0)

    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], marker="o", linewidth=1.5, color="tab:blue")
    ax.quiver(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        forward[:, 0],
        forward[:, 1],
        forward[:, 2],
        length=0.08,
        normalize=True,
        color="tab:red",
        label="camera +z",
    )
    ax.quiver(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        up[:, 0],
        up[:, 1],
        up[:, 2],
        length=0.05,
        normalize=True,
        color="tab:green",
        label="camera +y",
    )
    ax.set_title("Exported Camera Path (OpenCV camera-to-world)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _observed_export_pose(observation_frame) -> np.ndarray:
    x_vw = np.linalg.inv(np.asarray(observation_frame.X_WV, dtype=np.float64))
    x_vw[1:3, :] *= -1.0
    return x_vw


def _validate_export(
    paths: RunPaths,
    save_dir: Path,
    observations: Optional[Observations],
) -> list[str]:
    lines = ["## Export Validation"]

    if not paths.intrinsics_path.exists():
        lines.append("- Missing `intrinsics.txt`; cannot validate fake ScanNet export.")
        return lines

    intrinsics = _load_intrinsics(paths.intrinsics_path)
    k = intrinsics[:3, :3]
    lines.append(
        "- Intrinsics: "
        f"fx={_format_float(k[0, 0])}, fy={_format_float(k[1, 1])}, "
        f"cx={_format_float(k[0, 2])}, cy={_format_float(k[1, 2])}, "
        f"bx={_format_float(intrinsics[0, 3])}, by={_format_float(intrinsics[1, 3])}"
    )
    lines.append(f"- Last row exact identity: {np.allclose(intrinsics[3], np.array([0.0, 0.0, 0.0, 1.0]))}")

    pose_files = _load_pose_files(paths.pose_dir)
    if pose_files:
        poses = [np.loadtxt(path) for path in pose_files]
        pose_ids = [_numeric_stem(path) for path in pose_files]
        expected_ids = list(range(len(pose_ids)))
        lines.append(f"- Pose count: {len(poses)}")
        lines.append(f"- Sequential pose numbering: {pose_ids == expected_ids}")

        rot_errors = []
        determinants = []
        translation_norms = []
        for pose in poses:
            pose = np.asarray(pose, dtype=np.float64)
            rot_error, determinant = _rotation_stats(pose[:3, :3])
            rot_errors.append(rot_error)
            determinants.append(determinant)
            translation_norms.append(float(np.linalg.norm(pose[:3, 3])))

        lines.append(
            "- Pose rotation sanity: "
            f"max_orthogonality_error={_format_float(max(rot_errors))}, "
            f"det_range=[{_format_float(min(determinants))}, {_format_float(max(determinants))}]"
        )
        lines.append(
            "- Pose translation norm range: "
            f"[{_format_float(min(translation_norms))}, {_format_float(max(translation_norms))}] meters"
        )
        _plot_camera_path(poses, save_dir / "camera_path.png")
    else:
        lines.append("- Missing `pose/*.txt`; cannot validate camera path.")

    if paths.color_dir is not None and paths.color_dir.exists():
        color_files = _sorted_numbered_files(list(paths.color_dir.glob("*.jpg")))
        if color_files:
            first_color = imageio.imread(color_files[0])
            lines.append(f"- Color export: {len(color_files)} frames, sample shape={tuple(first_color.shape)}, dtype={first_color.dtype}")
        else:
            lines.append("- Color export directory exists but contains no `.jpg` frames.")
    else:
        lines.append("- Missing `color/*.jpg` export.")

    if paths.depth_dir is not None and paths.depth_dir.exists():
        depth_files = _sorted_numbered_files(list(paths.depth_dir.glob("*.png")))
        if depth_files:
            first_depth = imageio.imread(depth_files[0])
            depth_min = int(np.min(first_depth))
            depth_max = int(np.max(first_depth))
            non_zero = first_depth[first_depth > 0]
            non_zero_min = int(np.min(non_zero)) if non_zero.size else 0
            non_zero_max = int(np.max(non_zero)) if non_zero.size else 0
            lines.append(
                "- Depth export: "
                f"{len(depth_files)} frames, sample shape={tuple(first_depth.shape)}, dtype={first_depth.dtype}, "
                f"range=[{depth_min}, {depth_max}] mm, nonzero_range=[{non_zero_min}, {non_zero_max}] mm"
            )
        else:
            lines.append("- Depth export directory exists but contains no `.png` frames.")
    else:
        lines.append("- Missing `depth/*.png` export.")

    if paths.mesh_path is not None and paths.mesh_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(paths.mesh_path))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        bbox = mesh.get_axis_aligned_bounding_box()
        lines.append(
            "- Mesh export: "
            f"{len(vertices)} vertices, {len(triangles)} triangles, "
            f"bbox_min={np.round(bbox.min_bound, 4).tolist()}, bbox_max={np.round(bbox.max_bound, 4).tolist()}"
        )
    else:
        lines.append("- Missing `<scene>_vh_clean_2.ply` mesh export.")

    if observations is not None:
        lines.append("### Export Comparison Against Original DEG Observations")
        lines.append(f"- Original observation count: {len(observations.frames)}")

        if len(observations.frames) > 0:
            expected_intrinsics = np.eye(4, dtype=np.float64)
            expected_intrinsics[:3, :3] = np.asarray(observations.frames[0].K, dtype=np.float64)
            lines.append(
                "- Intrinsics max abs diff vs source: "
                f"{_format_float(float(np.max(np.abs(expected_intrinsics - intrinsics))))}"
            )

        if pose_files and len(observations.frames) == len(pose_files):
            pose_diffs = []
            for obs_frame, pose_file in zip(observations.frames, pose_files):
                actual_pose = np.asarray(np.loadtxt(pose_file), dtype=np.float64)
                expected_pose = _observed_export_pose(obs_frame)
                pose_diffs.append(float(np.max(np.abs(expected_pose - actual_pose))))
            lines.append(f"- Pose max abs diff vs source: {_format_float(max(pose_diffs))}")
        elif pose_files:
            lines.append(
                f"- Pose comparison skipped because exported pose count ({len(pose_files)}) "
                f"!= observation count ({len(observations.frames)})."
            )

        if paths.color_dir is not None and paths.color_dir.exists():
            color_files = _sorted_numbered_files(list(paths.color_dir.glob("*.jpg")))
            if len(color_files) == len(observations.frames):
                diffs = []
                for obs_frame, color_file in zip(observations.frames, color_files):
                    exported = imageio.imread(color_file)
                    source = np.clip(np.asarray(obs_frame.color) * 255.0, 0, 255).astype(np.uint8)
                    if exported.shape == source.shape:
                        diffs.append(float(np.mean(np.abs(exported.astype(np.float32) - source.astype(np.float32)))))
                if diffs:
                    lines.append(
                        "- Color mean absolute diff vs source after JPG export: "
                        f"{_format_float(float(np.mean(diffs)))} intensity levels"
                    )

        if paths.depth_dir is not None and paths.depth_dir.exists():
            depth_files = _sorted_numbered_files(list(paths.depth_dir.glob("*.png")))
            if len(depth_files) == len(observations.frames):
                depth_diffs = []
                for obs_frame, depth_file in zip(observations.frames, depth_files):
                    if obs_frame.depth is None:
                        continue
                    exported = imageio.imread(depth_file).astype(np.int64)
                    source = np.clip(np.asarray(obs_frame.depth) * 1000.0, 0, 65535).astype(np.uint16).astype(np.int64)
                    if exported.shape == source.shape:
                        depth_diffs.append(int(np.max(np.abs(exported - source))))
                if depth_diffs:
                    lines.append(f"- Depth max abs diff vs source after mm conversion: {max(depth_diffs)} mm")

    return lines


def _load_export_frames(paths: RunPaths) -> list[Frame]:
    if paths.color_dir is None or paths.depth_dir is None or paths.pose_dir is None:
        return []
    if not (paths.color_dir.exists() and paths.depth_dir.exists() and paths.pose_dir.exists() and paths.intrinsics_path.exists()):
        return []

    intrinsics = _load_intrinsics(paths.intrinsics_path)[:3, :3]
    color_files = _sorted_numbered_files(list(paths.color_dir.glob("*.jpg")))
    depth_files = _sorted_numbered_files(list(paths.depth_dir.glob("*.png")))
    pose_files = _sorted_numbered_files(list(paths.pose_dir.glob("*.txt")))
    if not color_files or len(color_files) != len(depth_files) or len(depth_files) != len(pose_files):
        return []

    frames: list[Frame] = []
    for color_file, depth_file, pose_file in zip(color_files, depth_files, pose_files):
        color = imageio.imread(color_file)
        depth_mm = imageio.imread(depth_file)
        pose = np.asarray(np.loadtxt(pose_file), dtype=np.float32)

        x_vw = pose.copy()
        x_vw[1:3, :] *= -1.0
        x_wv = np.linalg.inv(x_vw)

        frame = Frame(
            id=_numeric_stem(color_file),
            name=color_file.stem,
            color=torch.tensor(color.astype(np.float32) / 255.0, device="cuda"),
            X_WV=torch.tensor(x_wv.astype(np.float32)),
            K=torch.tensor(intrinsics.astype(np.float32)),
            depth=torch.tensor(depth_mm.astype(np.float32) / 1000.0, device="cuda"),
        )
        frames.append(frame)
    return frames


def _plot_prompt_cloud(prompt_xyz: np.ndarray, mesh: Optional[o3d.geometry.TriangleMesh], save_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if mesh is not None and mesh.has_vertices():
        vertices = np.asarray(mesh.vertices)
        if len(vertices) > 50000:
            rng = np.random.default_rng(42)
            vertices = vertices[rng.choice(len(vertices), size=50000, replace=False)]
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.2, c="lightgray", alpha=0.4)

    ax.scatter(prompt_xyz[:, 0], prompt_xyz[:, 1], prompt_xyz[:, 2], s=6.0, c="tab:red", alpha=0.95)
    ax.set_title("Initial Prompt Cloud Over Mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _save_points_overlay(image: np.ndarray, points: np.ndarray, save_path: Path, title: str, subtitle: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], s=24, c="tab:red", edgecolors="white", linewidths=0.6)
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _save_filter_overlay(
    image: np.ndarray,
    raw_points: np.ndarray,
    local_keep_points: np.ndarray,
    global_keep_points: np.ndarray,
    raw_mask_union: Optional[np.ndarray],
    kept_mask_union: Optional[np.ndarray],
    save_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    if raw_mask_union is not None:
        axes[0].imshow(np.ma.masked_where(~raw_mask_union, raw_mask_union), cmap="Blues", alpha=0.25)
    if len(raw_points) > 0:
        axes[0].scatter(raw_points[:, 0], raw_points[:, 1], s=18, c="tab:orange", edgecolors="white", linewidths=0.4)
    axes[0].set_title("Before Prompt Filtering")
    axes[0].axis("off")

    axes[1].imshow(image)
    if kept_mask_union is not None:
        axes[1].imshow(np.ma.masked_where(~kept_mask_union, kept_mask_union), cmap="Greens", alpha=0.25)
    if len(local_keep_points) > 0:
        axes[1].scatter(local_keep_points[:, 0], local_keep_points[:, 1], s=18, c="tab:blue", edgecolors="white", linewidths=0.4)
    if len(global_keep_points) > 0:
        axes[1].scatter(global_keep_points[:, 0], global_keep_points[:, 1], s=24, c="tab:green", edgecolors="white", linewidths=0.5)
    axes[1].set_title("After Local + Global Prompt Filtering")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _project_prompts(
    prompt_xyz: np.ndarray,
    color_image: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics_4x4: np.ndarray,
    pose_c2w_opencv: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_tensor = torch.tensor(prompt_xyz, device=device, dtype=torch.float32)
    depth_tensor = torch.tensor(depth_mm.astype(np.float64), device=device)
    intrinsics_tensor = torch.tensor(intrinsics_4x4.astype(np.float64), device=device)
    pose_tensor = torch.tensor(pose_c2w_opencv.astype(np.float64), device=device)
    projected_points, keep_idx = transform_pt_depth_scannet_torch(prompt_tensor, intrinsics_tensor, depth_tensor, pose_tensor, device)
    return projected_points.detach().cpu().numpy(), keep_idx.detach().cpu().numpy()


def _load_rgbd_pose_for_frame(paths: RunPaths, frame_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert paths.color_dir is not None and paths.depth_dir is not None and paths.pose_dir is not None
    color = imageio.imread(paths.color_dir / f"{frame_name}.jpg")
    depth = imageio.imread(paths.depth_dir / f"{frame_name}.png")
    pose = np.asarray(np.loadtxt(paths.pose_dir / f"{frame_name}.txt"), dtype=np.float64)
    return color, depth, pose


def _compute_global_keep_idx(prompt_xyz: np.ndarray, sam_output_scene_dir: Path, args: Args) -> torch.Tensor:
    dummy_predictor = SimpleNamespace(model=SimpleNamespace(mask_threshold=args.mask_threshold))
    filter_args = SimpleNamespace(
        device=args.device,
        pred_iou_thres=args.pred_iou_thres,
        stability_score_thres=args.stability_score_thres,
        box_nms_thres=args.box_nms_thres,
        keep_thres=args.keep_thres,
    )
    npy_files = _sorted_numbered_files(list((sam_output_scene_dir / "points_npy").glob("*.npy")))
    prompt_tensor = torch.tensor(prompt_xyz, device=args.device, dtype=torch.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        return prompt_filter(prompt_tensor, str(sam_output_scene_dir), [path.name for path in npy_files], dummy_predictor, filter_args)


def _filtered_stage1_data(
    scene_output_dir: Path,
    npy_name: str,
    global_keep_idx: torch.Tensor,
    mask_threshold: float,
    pred_iou_thres: float,
    stability_score_thres: float,
    box_nms_thres: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    from torchvision.ops import batched_nms

    points = torch.from_numpy(np.load(scene_output_dir / "points_npy" / npy_name)).to(device)
    iou_preds = torch.from_numpy(np.load(scene_output_dir / "iou_preds_npy" / npy_name)).to(device)
    masks = torch.from_numpy(np.load(scene_output_dir / "masks_npy" / npy_name)).to(device)
    corre = torch.from_numpy(np.load(scene_output_dir / "corre_3d_ins_npy" / npy_name)).to(device)

    data = MaskData(masks=masks, iou_preds=iou_preds, points=points, corre_3d_ins=corre)
    raw_points = data["points"].detach().cpu().numpy()
    raw_mask_union = None
    if len(data["masks"]) > 0:
        raw_mask_union = (data["masks"] > mask_threshold).any(dim=0).detach().cpu().numpy()

    if pred_iou_thres > 0.0:
        data.filter(data["iou_preds"] > pred_iou_thres)

    if len(data["masks"]) > 0 and stability_score_thres > 0.0:
        data["stability_score"] = calculate_stability_score(data["masks"], mask_threshold=mask_threshold, threshold_offset=1.0)
        data.filter(data["stability_score"] >= stability_score_thres)

    local_keep_points = np.zeros((0, 2), dtype=np.float32)
    global_keep_points = np.zeros((0, 2), dtype=np.float32)
    kept_mask_union = None

    if len(data["masks"]) > 0:
        binary_masks = data["masks"] > mask_threshold
        boxes = batched_mask_to_box(binary_masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            data["iou_preds"],
            torch.zeros_like(boxes[:, 0]),
            iou_threshold=box_nms_thres,
        )
        data.filter(keep_by_nms)
        local_keep_points = data["points"].detach().cpu().numpy()

        global_mask = torch.isin(data["corre_3d_ins"], global_keep_idx.to(data["corre_3d_ins"].device))
        data.filter(global_mask)
        if len(data["points"]) > 0:
            global_keep_points = data["points"].detach().cpu().numpy()
            kept_mask_union = (data["masks"] > mask_threshold).any(dim=0).detach().cpu().numpy()

    return raw_points, local_keep_points, global_keep_points, raw_mask_union, kept_mask_union


def _visualize_prompts(paths: RunPaths, save_dir: Path, args: Args) -> list[str]:
    lines = ["## Prompt And Projection Diagnostics"]

    if paths.scene_id is None:
        lines.append("- Could not determine `scene_id`; prompt diagnostics skipped.")
        return lines

    if paths.prompt_ply_path is None or not paths.prompt_ply_path.exists():
        lines.append("- Missing `init_prompt/<scene>.ply`; prompt diagnostics skipped.")
        return lines

    prompt_xyz, _ = load_ply(str(paths.prompt_ply_path))
    lines.append(f"- Prompt cloud: {len(prompt_xyz)} initial 3D prompts from `{paths.prompt_ply_path.name}`.")

    mesh = o3d.io.read_triangle_mesh(str(paths.mesh_path)) if paths.mesh_path is not None and paths.mesh_path.exists() else None
    _plot_prompt_cloud(prompt_xyz, mesh, save_dir / "prompt_cloud_overview.png")

    if args.open3d:
        geometries: list[o3d.geometry.Geometry] = []
        if mesh is not None:
            mesh_vis = o3d.geometry.TriangleMesh(mesh)
            mesh_vis.paint_uniform_color([0.8, 0.8, 0.8])
            geometries.append(mesh_vis)
        prompt_pcd = o3d.geometry.PointCloud()
        prompt_pcd.points = o3d.utility.Vector3dVector(prompt_xyz)
        prompt_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(prompt_pcd)
        if geometries:
            try:
                o3d.visualization.draw_geometries(geometries, window_name="SAMPro3D Prompt Cloud")
            except Exception:
                logger.exception("Failed to open Open3D prompt viewer")

    if not paths.intrinsics_path.exists():
        lines.append("- Missing `intrinsics.txt`; cannot project prompts into images.")
        return lines

    if paths.color_dir is None or paths.depth_dir is None or paths.pose_dir is None:
        lines.append("- Missing exported color/depth/pose directories; projection overlays skipped.")
        return lines

    color_files = _sorted_numbered_files(list(paths.color_dir.glob("*.jpg"))) if paths.color_dir.exists() else []
    depth_files = _sorted_numbered_files(list(paths.depth_dir.glob("*.png"))) if paths.depth_dir.exists() else []
    pose_files = _sorted_numbered_files(list(paths.pose_dir.glob("*.txt"))) if paths.pose_dir.exists() else []
    if not color_files or len(color_files) != len(depth_files) or len(depth_files) != len(pose_files):
        lines.append("- Need matching `color`, `depth`, and `pose` exports to generate prompt overlays.")
        return lines

    intrinsics = _load_intrinsics(paths.intrinsics_path)
    sample_indices = _sample_indices(len(color_files), args.max_frames)
    projection_counts: list[str] = []
    for idx in sample_indices:
        frame_name = color_files[idx].stem
        color, depth, pose = _load_rgbd_pose_for_frame(paths, frame_name)
        projected_points, keep_idx = _project_prompts(prompt_xyz, color, depth, intrinsics, pose, args.device)
        projection_counts.append(f"{frame_name}:{len(keep_idx)}/{len(prompt_xyz)}")
        _save_points_overlay(
            color,
            projected_points,
            save_dir / f"prompt_projection_{frame_name}.png",
            f"Projected Prompts For Frame {frame_name}",
            subtitle=f"visible prompts: {len(keep_idx)} / {len(prompt_xyz)}",
        )

    lines.append(f"- Prompt reprojection visible counts by sampled frame: {', '.join(projection_counts)}")

    if paths.sam_output_scene_dir is None or not paths.sam_output_scene_dir.exists():
        lines.append("- Missing `sam_output/<scene>`; stage-1 filter diagnostics skipped.")
        return lines

    required_dirs = [
        paths.sam_output_scene_dir / "points_npy",
        paths.sam_output_scene_dir / "iou_preds_npy",
        paths.sam_output_scene_dir / "masks_npy",
        paths.sam_output_scene_dir / "corre_3d_ins_npy",
    ]
    if not all(path.exists() for path in required_dirs):
        lines.append("- `sam_output` is incomplete; prompt filter diagnostics skipped.")
        return lines

    global_keep_idx = _compute_global_keep_idx(prompt_xyz, paths.sam_output_scene_dir, args)
    lines.append(f"- Global prompt filter kept {len(global_keep_idx)} / {len(prompt_xyz)} prompt IDs.")

    stage1_files = _sorted_numbered_files(list((paths.sam_output_scene_dir / "points_npy").glob("*.npy")))
    sample_indices = _sample_indices(len(stage1_files), args.max_frames)
    for idx in sample_indices:
        npy_name = stage1_files[idx].name
        frame_name = stage1_files[idx].stem
        if paths.color_dir is None:
            continue
        color = imageio.imread(paths.color_dir / f"{frame_name}.jpg")
        raw_points, local_keep_points, global_keep_points, raw_mask_union, kept_mask_union = _filtered_stage1_data(
            paths.sam_output_scene_dir,
            npy_name,
            global_keep_idx,
            mask_threshold=args.mask_threshold,
            pred_iou_thres=args.pred_iou_thres,
            stability_score_thres=args.stability_score_thres,
            box_nms_thres=args.box_nms_thres,
            device=args.device,
        )
        _save_filter_overlay(
            color,
            raw_points,
            local_keep_points,
            global_keep_points,
            raw_mask_union,
            kept_mask_union,
            save_dir / f"stage1_prompt_filter_{frame_name}.png",
            f"Stage-1 Prompt Filtering For Frame {frame_name}",
        )

    return lines


def _save_colored_mesh(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray, save_path: Path) -> None:
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.vertex_colors = o3d.utility.Vector3dVector(_random_colors_for_labels(labels))
    o3d.io.write_triangle_mesh(str(save_path), mesh_copy)


def _save_instance_mask_overlay(frame: Frame, mask: np.ndarray, save_path: Path, title: str) -> None:
    colors = _random_colors_for_labels(mask.ravel()).reshape(mask.shape + (3,))
    rgb = np.clip(frame.color.detach().cpu().numpy(), 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].imshow(colors, alpha=0.45)
    axes[1].set_title("Back-Projected Instance IDs")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _autodetect_post_floor(pred_dir: Path, scene_id: str) -> bool:
    return (pred_dir / f"{scene_id}_seg_floor.npy").exists() and not (pred_dir / f"{scene_id}_seg.npy").exists()


def _visualize_predictions(
    paths: RunPaths,
    save_dir: Path,
    scene: Optional[SceneSetup],
    frames: list[Frame],
    args: Args,
) -> list[str]:
    lines = ["## Prediction Diagnostics"]

    if paths.scene_id is None:
        lines.append("- Could not determine `scene_id`; prediction diagnostics skipped.")
        return lines
    if paths.mesh_path is None or not paths.mesh_path.exists():
        lines.append("- Missing mesh export; cannot color predictions on geometry.")
        return lines
    if not paths.pred_dir.exists():
        lines.append("- Missing `final_pred`; prediction diagnostics skipped.")
        return lines

    post_floor = args.post_floor if args.post_floor is not None else _autodetect_post_floor(paths.pred_dir, paths.scene_id)
    try:
        raw_labels = _load_sampro3d_labels(paths.pred_dir, paths.scene_id, post_floor)
    except FileNotFoundError as exc:
        lines.append(f"- {exc}")
        return lines

    mesh = o3d.io.read_triangle_mesh(str(paths.mesh_path))
    vertices = np.asarray(mesh.vertices)
    if len(vertices) != len(raw_labels):
        lines.append(
            f"- Label count ({len(raw_labels)}) does not match mesh vertex count ({len(vertices)}); "
            "prediction visualizations skipped."
        )
        return lines

    _save_colored_mesh(mesh, raw_labels, save_dir / "mesh_segmented_raw.ply")
    lines.append(f"- Raw normalized labels: {len(np.unique(raw_labels[raw_labels > 0]))} foreground instances.")

    if args.open3d:
        try:
            mesh_vis = o3d.io.read_triangle_mesh(str(save_dir / "mesh_segmented_raw.ply"))
            o3d.visualization.draw_geometries([mesh_vis], window_name="SAMPro3D Raw Labels")
        except Exception:
            logger.exception("Failed to open Open3D raw label mesh viewer")

    if scene is None:
        lines.append("- `scene_path` not provided; workspace/table filtering comparison skipped.")
        return lines
    if not frames:
        lines.append("- Exported RGB-D frames unavailable; back-projected prediction masks skipped.")
        return lines

    filtered_labels = _filter_labels_by_workspace(vertices, raw_labels, get_workspace_voxels(scene))
    instance_groups = _render_instance_id_masks(mesh, filtered_labels, frames) or {}
    valid_ids = np.array([label for label in np.unique(filtered_labels) if label > 0], dtype=np.int32)

    if len(valid_ids) > 0:
        frame_counts = {label: 0 for label in valid_ids}
        for mask in instance_groups.values():
            present = np.unique(mask[mask > 0])
            for label in present:
                frame_counts[int(label)] += 1
        valid_ids = np.array([label for label in valid_ids if frame_counts[int(label)] >= 3], dtype=np.int32)
        for name, mask in instance_groups.items():
            instance_groups[name] = np.where(np.isin(mask, valid_ids), mask, 0)

    table_id = determine_table_instance_id(frames, instance_groups, scene.ground_plane, valid_ids)
    if table_id > 0:
        valid_ids = valid_ids[valid_ids != table_id]
        for name, mask in instance_groups.items():
            instance_groups[name] = np.where(mask == table_id, 0, mask)

    final_filtered_labels = filtered_labels.copy()
    final_filtered_labels[~np.isin(final_filtered_labels, valid_ids)] = 0
    _save_colored_mesh(mesh, final_filtered_labels, save_dir / "mesh_segmented_filtered.ply")
    lines.append(
        f"- Filtered labels: {len(np.unique(final_filtered_labels[final_filtered_labels > 0]))} foreground instances "
        f"after workspace, frame-count, and table filtering."
    )

    sample_indices = _sample_indices(len(frames), args.max_frames)
    for idx in sample_indices:
        frame = frames[idx]
        mask = instance_groups.get(frame.name)
        if mask is None:
            continue
        _save_instance_mask_overlay(
            frame,
            mask,
            save_dir / f"predicted_pixel_masks_{frame.name}.png",
            title=f"Back-Projected Predictions For Frame {frame.name}",
        )

    if args.open3d:
        try:
            mesh_vis = o3d.io.read_triangle_mesh(str(save_dir / "mesh_segmented_filtered.ply"))
            o3d.visualization.draw_geometries([mesh_vis], window_name="SAMPro3D Filtered Labels")
        except Exception:
            logger.exception("Failed to open Open3D filtered label mesh viewer")

    return lines


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)-18s %(levelname)-8s %(message)s")
    args = tyro.cli(Args)

    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    save_dir = (args.save_dir.resolve() if args.save_dir is not None else output_dir / "visualize")
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = _resolve_run_paths(output_dir)
    observations = Observations.load(args.observations_path) if args.observations_path is not None else None
    scene = SceneSetup.load(args.scene_path) if args.scene_path is not None else None
    frames = _load_export_frames(paths)

    report_lines = [
        "# SAMPro3D Visualization Summary",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Save directory: `{save_dir}`",
        f"- Scene id: `{paths.scene_id}`",
        f"- Loaded exported RGB-D frames: {len(frames)}",
        "",
    ]

    inventory_lines, _ = _inventory(paths)
    report_lines.extend(inventory_lines)
    report_lines.append("")

    if args.stage in ("all", "export"):
        report_lines.extend(_validate_export(paths, save_dir, observations))
        report_lines.append("")

    if args.stage in ("all", "prompts"):
        report_lines.extend(_visualize_prompts(paths, save_dir, args))
        report_lines.append("")

    if args.stage in ("all", "predictions"):
        report_lines.extend(_visualize_predictions(paths, save_dir, scene, frames, args))
        report_lines.append("")

    report = "\n".join(report_lines).strip() + "\n"
    _write_text(save_dir / "summary.md", report)
    print(report)


if __name__ == "__main__":
    run()
