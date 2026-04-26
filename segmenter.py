"""SAMPro3D DEG bridge module.

Adapts SAMPro3D's two-stage pipeline to the DEG external segmenter contract.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch

from initializerdefs import (
    InstanceMaskObjectsDef,
    ObjectSegmentations,
    ObservationFrame,
    Observations,
    SceneSetup,
)
from psdframe import Frame


logger = logging.getLogger("sampro3d-segmenter")

DEFAULT_SEED = 42
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
SAM_CHECKPOINT_DEFAULTS = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


def get_dataset_frame_from_observation_frame(observation_frame: ObservationFrame) -> Frame:
    return Frame(
        id=observation_frame.id,
        name=observation_frame.name,
        color=torch.tensor(observation_frame.color).cuda(),
        X_WV=torch.tensor(observation_frame.X_WV),
        K=torch.tensor(observation_frame.K),
        depth=(torch.tensor(observation_frame.depth).cuda() if observation_frame.depth is not None else None),
    )


def _to_cam_open3d(frame: Frame) -> o3d.camera.PinholeCameraParameters:
    intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy)
    extrinsic = frame.X_VW_opencv.cpu().numpy()
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic
    return camera


def _post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int = 1000) -> o3d.geometry.TriangleMesh:
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, _ = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    logger.info("mesh vertices raw %d -> post %d", len(mesh.vertices), len(mesh_0.vertices))
    return mesh_0


@torch.no_grad()
def _extract_mesh_bounded(
    frames: List[Frame],
    voxel_size: float = 0.004,
    sdf_trunc: float = 0.02,
    depth_trunc: float = 3,
) -> o3d.geometry.TriangleMesh:
    logger.info(
        "TSDF integration: voxel_size=%.4f  sdf_trunc=%.4f  depth_trunc=%.2f",
        voxel_size,
        sdf_trunc,
        depth_trunc,
    )
    for frame in frames:
        assert frame.depth is not None and frame.color is not None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame in frames:
        rgb = frame.color.cpu().numpy()
        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        assert depth is not None

        ci = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        di = o3d.geometry.Image(depth)
        cam = _to_cam_open3d(frame)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(ci, di, depth_trunc=depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.0)
        volume.integrate(rgbd, intrinsic=cam.intrinsic, extrinsic=cam.extrinsic)

    return volume.extract_triangle_mesh()


def _extract_mesh_bounded_with_res(frames: List[Frame], depth_trunc: float = 2, mesh_res: int = 1024) -> o3d.geometry.TriangleMesh:
    voxel_size = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel_size
    raw_mesh = _extract_mesh_bounded(frames, voxel_size, sdf_trunc, depth_trunc)
    return _post_process_mesh(raw_mesh, cluster_to_keep=50)


def _erode_voxel_grid_xy(voxel_grid: o3d.geometry.VoxelGrid, layers: int) -> o3d.geometry.VoxelGrid:
    if layers <= 0 or not voxel_grid.has_voxels():
        return voxel_grid

    voxel_indices = [tuple(int(idx) for idx in voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    xy_occupied = {(x, y) for x, y, _ in voxel_indices}

    for _ in range(layers):
        if not xy_occupied:
            break
        prev_xy = xy_occupied
        xy_occupied = {
            (x, y) for (x, y) in prev_xy if ((x - 1, y) in prev_xy and (x + 1, y) in prev_xy and (x, y - 1) in prev_xy and (x, y + 1) in prev_xy)
        }

    for voxel_index in voxel_indices:
        if (voxel_index[0], voxel_index[1]) not in xy_occupied:
            voxel_grid.remove_voxel(voxel_index)

    return voxel_grid


def get_workspace_voxels(scene: SceneSetup, shrink_xy_m: float = 0.04) -> o3d.geometry.VoxelGrid:
    table_xyz = scene.ground_gaussians.xyz
    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_pcd_extruded = np.array(table_xyz).copy()

    desired_height = 1.0
    below_table_height = 0.10
    voxel_size = 0.02
    iters = int(np.ceil(desired_height / voxel_size))
    for i in range(iters):
        new_points = table_xyz + table_normal * voxel_size * i
        table_pcd_extruded = np.append(table_pcd_extruded, new_points, axis=0)

    below_table_iters = int(np.ceil(below_table_height / voxel_size))
    for i in range(below_table_iters):
        table_pcd_extruded = np.append(table_pcd_extruded, table_xyz - table_normal * voxel_size * (i + 1), axis=0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    layers = max(0, int(np.round(shrink_xy_m / voxel_grid.voxel_size)))
    return _erode_voxel_grid_xy(voxel_grid, layers)


def _crop_mesh_to_workspace_bbox(
    mesh: o3d.geometry.TriangleMesh,
    workspace_voxels: o3d.geometry.VoxelGrid,
    padding_m: float = 0.2,
) -> o3d.geometry.TriangleMesh:
    voxel_size = float(workspace_voxels.voxel_size)
    origin = np.asarray(workspace_voxels.origin, dtype=np.float32)
    voxels = workspace_voxels.get_voxels()
    if len(voxels) == 0:
        return mesh

    indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
    min_corner = origin + indices.min(axis=0) * voxel_size - padding_m
    max_corner = origin + (indices.max(axis=0) + 1.0) * voxel_size + padding_m
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
    return mesh.crop(aabb)


def _crop_mesh_to_workspace(
    mesh: o3d.geometry.TriangleMesh,
    workspace_voxels: o3d.geometry.VoxelGrid,
) -> o3d.geometry.TriangleMesh:
    if not mesh.has_triangles() or not mesh.has_vertices() or not workspace_voxels.has_voxels():
        return mesh

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        return mesh

    in_workspace = np.asarray(
        workspace_voxels.check_if_included(o3d.utility.Vector3dVector(vertices)),
        dtype=bool,
    )
    keep_triangles = in_workspace[triangles].all(axis=1)

    if keep_triangles.all():
        return mesh

    mesh.remove_triangles_by_mask(~keep_triangles)
    mesh.remove_unreferenced_vertices()
    return mesh


def _filter_labels_by_workspace(
    vertices: np.ndarray,
    labels: np.ndarray,
    workspace_voxels: o3d.geometry.VoxelGrid,
) -> np.ndarray:
    labels = labels.copy()
    pcd = o3d.utility.Vector3dVector(vertices)
    valid_mask = np.array(workspace_voxels.check_if_included(pcd))

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl <= 0:
            continue
        seg_mask = labels == lbl
        total = np.sum(seg_mask)
        inside = np.sum(valid_mask & seg_mask)
        if total > 0 and inside / total < 0.9:
            labels[seg_mask] = 0

    return labels


def _triangle_labels_from_vertices(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> np.ndarray:
    triangles = np.asarray(mesh.triangles)
    tri_labels = labels[triangles]
    a = tri_labels[:, 0]
    b = tri_labels[:, 1]
    c = tri_labels[:, 2]
    return np.where((a == b) | (a == c), a, np.where(b == c, b, a))


def _render_instance_id_masks(
    mesh: o3d.geometry.TriangleMesh,
    labels: np.ndarray,
    frames: List[Frame],
) -> Optional[Dict[str, np.ndarray]]:
    def _to_legacy_mesh(input_mesh):
        if isinstance(input_mesh, o3d.geometry.TriangleMesh):
            return input_mesh
        if hasattr(input_mesh, "to_legacy"):
            try:
                return input_mesh.to_legacy()
            except Exception:
                pass
        legacy = o3d.geometry.TriangleMesh()
        legacy.vertices = o3d.utility.Vector3dVector(np.asarray(input_mesh.vertices))
        legacy.triangles = o3d.utility.Vector3iVector(np.asarray(input_mesh.triangles))
        return legacy

    def _to_tensor_mesh(input_mesh: o3d.geometry.TriangleMesh) -> "o3d.t.geometry.TriangleMesh":
        return o3d.t.geometry.TriangleMesh.from_legacy(input_mesh)

    def _raycast_instance_id_masks(
        mesh_legacy: o3d.geometry.TriangleMesh,
        tri_labels: np.ndarray,
        frames: List[Frame],
    ) -> Dict[str, np.ndarray]:
        scene = o3d.t.geometry.RaycastingScene()
        tmesh = _to_tensor_mesh(mesh_legacy)
        scene.add_triangles(tmesh)

        result: Dict[str, np.ndarray] = {}
        for frame in frames:
            h, w = frame.h, frame.w
            k = frame.K.cpu().numpy()
            fx = float(k[0, 0])
            fy = float(k[1, 1])
            cx = float(k[0, 2])
            cy = float(k[1, 2])

            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            u = u + 0.5
            v = v + 0.5
            dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
            dirs_cam = dirs_cam.reshape(-1, 3)
            dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

            x_vw = frame.X_VW_opencv.cpu().numpy()
            x_wv = np.linalg.inv(x_vw)
            r = x_wv[:3, :3]
            t = x_wv[:3, 3]
            dirs_world = dirs_cam @ r.T
            origins = np.broadcast_to(t, dirs_world.shape)

            rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
            ans = scene.cast_rays(o3d.core.Tensor(rays))
            prim_ids = ans["primitive_ids"].numpy().reshape(h, w)

            mask = np.zeros((h, w), dtype=np.int32)
            if np.issubdtype(prim_ids.dtype, np.unsignedinteger):
                invalid = np.iinfo(prim_ids.dtype).max
                hit = prim_ids != invalid
            else:
                hit = prim_ids >= 0
            if np.any(hit):
                prim_ids_valid = prim_ids[hit].astype(np.int64)
                mask[hit] = tri_labels[prim_ids_valid]
            result[frame.name] = mask

        return result

    mesh_legacy = _to_legacy_mesh(copy.deepcopy(mesh))
    tri_labels = _triangle_labels_from_vertices(mesh_legacy, labels)
    return _raycast_instance_id_masks(mesh_legacy, tri_labels, frames)


def _get_instance_id_mask_for_frame(instance_id: int, masks: Dict[str, np.ndarray], frame: Frame) -> torch.Tensor:
    frame_mask = masks[frame.name]
    instance_mask = frame_mask == instance_id
    return torch.tensor(instance_mask, device=frame.color.device, dtype=torch.bool)


def determine_table_instance_id(
    frames: List[Frame],
    masks: Dict[str, np.ndarray],
    table_plane: Tuple[float, float, float, float],
    object_ids: np.ndarray,
) -> int:
    instance_ids = object_ids
    if len(instance_ids) == 0:
        return -1

    table_instance_candidates: List[int] = []
    table_instance_counts: List[int] = []

    for frame in frames:
        assert frame.depth is not None
        h, w = frame.depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=frame.depth.device),
            torch.arange(w, device=frame.depth.device),
            indexing="ij",
        )
        valid_mask = frame.depth > 0
        z = frame.depth
        x_world = (x - frame.cx) * z / frame.fl_x
        y_world = (y - frame.cy) * z / frame.fl_y
        points = torch.stack([x_world, y_world, z, torch.ones_like(z)], dim=0)
        points = frame.X_WV_opencv.cuda() @ points.reshape(4, -1)
        points = points.reshape(4, h, w)

        a, b, c, d = table_plane
        plane_dist = (a * points[0] + b * points[1] + c * points[2] + d) / math.sqrt(a * a + b * b + c * c)
        table_mask = torch.abs(plane_dist) < 0.02
        table_mask = table_mask & valid_mask

        for instance_id in instance_ids:
            inst_id_val = int(instance_id.item()) if hasattr(instance_id, "item") else int(instance_id)
            instance_mask = _get_instance_id_mask_for_frame(inst_id_val, masks, frame)
            instance_mask_valid = instance_mask & valid_mask
            instance_mask_near_table = instance_mask & table_mask
            valid_count = instance_mask_valid.sum()
            if valid_count > 0 and instance_mask_near_table.sum() / valid_count > 0.7:
                table_instance_candidates.append(inst_id_val)
                table_instance_counts.append(instance_mask_near_table.sum().item())

    if len(table_instance_candidates) == 0:
        logger.warning("No table candidates found")
        return -1

    best_idx = int(np.argmax(table_instance_counts))
    return table_instance_candidates[best_idx]


def _sanitize_scene_id(raw_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in raw_id)
    cleaned = cleaned.strip("_")
    return cleaned or "scene"


def _ensure_sam_checkpoint(model_type: str, checkpoint_path: Optional[Path]) -> Path:
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    default_name = SAM_CHECKPOINT_DEFAULTS.get(model_type, f"sam_{model_type}.pth")
    cache_dir = Path.home() / ".cache" / "sampro3d"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / default_name

    if cached.exists():
        logger.info("Using cached SAM checkpoint: %s", cached)
        return cached

    url = SAM_CHECKPOINT_URLS.get(model_type)
    if url is None:
        raise ValueError(f"Unknown SAM model type '{model_type}'. Cannot auto-download checkpoint.")

    logger.info("Downloading SAM checkpoint (%s) to %s ...", model_type, cached)
    try:
        urlretrieve(url, str(cached))
    except Exception as exc:
        raise RuntimeError(f"Failed to download SAM checkpoint from {url}: {exc}")

    if not cached.exists():
        raise RuntimeError("Checkpoint download completed but file is missing.")

    logger.info("Download complete.")
    return cached


def _write_scannet_temp_dataset(
    frames: List[Frame],
    scene_id: str,
    mesh: o3d.geometry.TriangleMesh,
    work_root: Path,
) -> Path:
    """Write ScanNet-style temp dataset expected by SAMPro3D.

    Returns the path to the dataset root (e.g. <work_root>/dataset/scannet).
    """
    dataset_root = work_root / "dataset" / "scannet"
    scene_dir = dataset_root / scene_id
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"

    for path in (color_dir, depth_dir, pose_dir):
        path.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError("No frames to export")

    # SAMPro3D expects a 4x4 intrinsics matrix
    k = frames[0].K.cpu().numpy()
    k4 = np.eye(4, dtype=np.float64)
    k4[:3, :3] = k
    np.savetxt(dataset_root / "intrinsics.txt", k4, fmt="%.8f")

    for idx, frame in enumerate(frames):
        frame_id = str(idx)
        color = (frame.color.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        imageio.imwrite(color_dir / f"{frame_id}.jpg", color)

        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        if depth is None:
            raise RuntimeError("Depth is required for SAMPro3D pipeline")
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        imageio.imwrite(depth_dir / f"{frame_id}.png", depth_mm)

        # SAMPro3D expects camera-to-world poses (it inverts to get world_to_camera)
        pose = frame.X_VW_opencv.cpu().numpy()
        np.savetxt(pose_dir / f"{frame_id}.txt", pose, fmt="%.8f")

    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh)

    logger.info("Wrote temp ScanNet dataset to %s (%d frames)", dataset_root, len(frames))
    return dataset_root


def _run_sampro3d_pipeline(
    dataset_root: Path,
    scene_id: str,
    work_root: Path,
    checkpoint: Path,
    model_type: str,
    device: str,
    voxel_size: float,
    pred_iou_thres: float,
    stability_score_thres: float,
    box_nms_thres: float,
    keep_thres: float,
    post_floor: bool,
    scene_ht_thres: float,
    scene_inter_thres: float,
    scene_dist_thres: float,
) -> Path:
    """Run SAMPro3D stage 1 + stage 2. Returns path to final_pred directory."""
    project_root = Path(__file__).parent
    prompt_path = work_root / "init_prompt"
    sam_output_path = work_root / "sam_output"
    pred_path = work_root / "final_pred"
    vis_path = work_root / "output_vis"

    for p in (prompt_path, sam_output_path, pred_path, vis_path):
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    # Stage 1: 3D prompt proposal
    stage1_cmd = [
        sys.executable,
        str(project_root / "3d_prompt_proposal.py"),
        "--data_path",
        str(dataset_root),
        "--scene_name",
        scene_id,
        "--prompt_path",
        str(prompt_path),
        "--sam_output_path",
        str(sam_output_path),
        "--voxel_size",
        str(voxel_size),
        "--model_type",
        model_type,
        "--sam_checkpoint",
        str(checkpoint),
        "--device",
        device,
    ]
    logger.info("Running SAMPro3D stage 1: %s", " ".join(stage1_cmd))
    result = subprocess.run(stage1_cmd, cwd=str(project_root), env=env, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SAMPro3D stage 1 failed with exit code {result.returncode}")

    # Stage 2: main segmentation
    stage2_cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "--data_path",
        str(dataset_root),
        "--scene_name",
        scene_id,
        "--prompt_path",
        str(prompt_path),
        "--sam_output_path",
        str(sam_output_path),
        "--pred_path",
        str(pred_path),
        "--output_vis_path",
        str(vis_path),
        "--model_type",
        model_type,
        "--sam_checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--pred_iou_thres",
        str(pred_iou_thres),
        "--stability_score_thres",
        str(stability_score_thres),
        "--box_nms_thres",
        str(box_nms_thres),
        "--keep_thres",
        str(keep_thres),
        "--post_floor",
        str(post_floor),
        "--scene_ht_thres",
        str(scene_ht_thres),
        "--scene_inter_thres",
        str(scene_inter_thres),
        "--scene_dist_thres",
        str(scene_dist_thres),
    ]
    logger.info("Running SAMPro3D stage 2: %s", " ".join(stage2_cmd))
    result = subprocess.run(stage2_cmd, cwd=str(project_root), env=env, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SAMPro3D stage 2 failed with exit code {result.returncode}")

    return pred_path


def _load_sampro3d_labels(pred_path: Path, scene_id: str, post_floor: bool) -> np.ndarray:
    """Load per-point instance labels from SAMPro3D output.

    SAMPro3D uses num_to_natural which maps unique non--1 values to 0,1,2,...
    and keeps -1 as -1. We remap -1 -> 0 and shift all others by +1 so that
    0 becomes background in DEG convention.
    """
    if post_floor:
        pred_file = pred_path / f"{scene_id}_seg_floor.npy"
    else:
        pred_file = pred_path / f"{scene_id}_seg.npy"

    if not pred_file.exists():
        raise FileNotFoundError(f"SAMPro3D prediction file not found: {pred_file}")

    labels = np.load(pred_file)
    logger.info("Loaded SAMPro3D labels: %s, unique=%s", labels.shape, np.unique(labels))

    # Normalize: SAMPro3D uses num_to_natural which maps:
    #   -1 = background, 0,1,2... = instance IDs
    # DEG expects 0 = background, 1,2,3... = instance IDs.
    labels = labels.copy()
    unique_values = np.unique(labels[labels != -1])
    if len(unique_values) == 0:
        return np.zeros_like(labels)
    mapping = np.zeros(np.max(unique_values) + 2, dtype=labels.dtype)
    mapping[unique_values + 1] = np.arange(len(unique_values)) + 1
    labels = mapping[labels + 1]

    return labels


def initialize_scene(
    observations: Observations,
    scene: SceneSetup,
    intermediate_outputs_path: Optional[Path] = None,
    device: str = "cuda",
    model_type: str = "vit_h",
    sam_checkpoint: Optional[Path] = None,
    voxel_size: float = 0.2,
    pred_iou_thres: float = 0.7,
    stability_score_thres: float = 0.6,
    box_nms_thres: float = 0.8,
    keep_thres: float = 0.4,
    post_floor: bool = False,
    scene_ht_thres: float = 0.08,
    scene_inter_thres: float = 0.4,
    scene_dist_thres: float = 0.01,
) -> ObjectSegmentations:
    """Run SAMPro3D on DEG observations and return ObjectSegmentations."""
    frames = [get_dataset_frame_from_observation_frame(f) for f in observations.frames]
    if not frames:
        raise ValueError("No frames in observations")

    # Reconstruct TSDF mesh
    logger.info("Reconstructing TSDF mesh from %d frames ...", len(frames))
    mesh = _extract_mesh_bounded_with_res(frames, depth_trunc=2, mesh_res=1024)

    # Build workspace voxels and crop mesh
    logger.info("Building workspace voxels ...")
    workspace_voxels = get_workspace_voxels(scene)
    mesh = _crop_mesh_to_workspace_bbox(mesh, workspace_voxels)
    mesh = _crop_mesh_to_workspace(mesh, workspace_voxels)
    if not mesh.has_triangles() or len(np.asarray(mesh.triangles)) == 0:
        raise RuntimeError("Mesh is empty after workspace cropping")

    # Ensure checkpoint
    checkpoint = _ensure_sam_checkpoint(model_type, sam_checkpoint)

    # Prepare temp work directory
    if intermediate_outputs_path is not None:
        work_root = intermediate_outputs_path / "sampro3d_work"
    else:
        work_root = Path(tempfile.mkdtemp(prefix="sampro3d_"))
    work_root.mkdir(parents=True, exist_ok=True)

    scene_id = _sanitize_scene_id(observations.id or "scene")

    # Write ScanNet-style temp dataset
    logger.info("Writing temp ScanNet dataset ...")
    dataset_root = _write_scannet_temp_dataset(frames, scene_id, mesh, work_root)

    # Run SAMPro3D pipeline
    pred_path = _run_sampro3d_pipeline(
        dataset_root=dataset_root,
        scene_id=scene_id,
        work_root=work_root,
        checkpoint=checkpoint,
        model_type=model_type,
        device=device,
        voxel_size=voxel_size,
        pred_iou_thres=pred_iou_thres,
        stability_score_thres=stability_score_thres,
        box_nms_thres=box_nms_thres,
        keep_thres=keep_thres,
        post_floor=post_floor,
        scene_ht_thres=scene_ht_thres,
        scene_inter_thres=scene_inter_thres,
        scene_dist_thres=scene_dist_thres,
    )

    # Load vertex labels
    vertex_labels = _load_sampro3d_labels(pred_path, scene_id, post_floor)

    # Filter by workspace (zero out labels for segments mostly outside workspace)
    vertices = np.asarray(mesh.vertices)
    vertex_labels = _filter_labels_by_workspace(vertices, vertex_labels, workspace_voxels)

    # Render per-frame instance masks via raycasting
    logger.info("Rendering per-frame instance masks ...")
    instance_groups = _render_instance_id_masks(mesh, vertex_labels, frames)
    if instance_groups is None:
        instance_groups = {}

    # Frame-count threshold: keep only labels visible in >= 3 frames
    unique_labels = np.unique(vertex_labels)
    valid_ids = np.array([lbl for lbl in unique_labels if lbl > 0])
    if len(valid_ids) > 0:
        label_frame_counts = {lbl: 0 for lbl in valid_ids}
        for name in instance_groups:
            for lbl in valid_ids:
                if np.any(instance_groups[name] == lbl):
                    label_frame_counts[lbl] += 1
        for lbl in valid_ids:
            if label_frame_counts[lbl] < 3:
                logger.info("Removing label %d (visible in %d frames)", lbl, label_frame_counts[lbl])
                valid_ids = valid_ids[valid_ids != lbl]
                for name in instance_groups:
                    instance_groups[name][instance_groups[name] == lbl] = 0

    # Remove table instance
    table_id = determine_table_instance_id(frames, instance_groups, scene.ground_plane, valid_ids)
    logger.info("Table instance id: %d", table_id)
    if table_id > 0:
        valid_ids = valid_ids[valid_ids != table_id]
        for name in instance_groups:
            instance_groups[name][instance_groups[name] == table_id] = 0

    # Final zero-out of any remaining invalid IDs
    for name in instance_groups:
        instance_groups[name][~np.isin(instance_groups[name], valid_ids)] = 0

    # Build InstanceMaskObjectsDef
    frame_ids: List[int] = []
    pixel_masks: List[np.ndarray] = []
    for obs_frame in observations.frames:
        frame_ids.append(obs_frame.id)
        mask = instance_groups.get(
            obs_frame.name,
            np.zeros((frames[0].h, frames[0].w), dtype=np.int32),
        )
        pixel_masks.append(mask)

    instance_mask_objects = InstanceMaskObjectsDef(
        frame_ids=frame_ids,
        pixel_object_ids=pixel_masks,
    )

    logger.info("Initialized %d objects (after table removal)", len(valid_ids))
    return ObjectSegmentations(object_segmentations=instance_mask_objects)
