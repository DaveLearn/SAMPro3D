# SAMPro3D DEG Integration Plan

This document describes the concrete steps needed to integrate `SAMPro3D` as a compatible DEG external segmenter.

## Goal

Make `dependencies/SAMPro3D` behave like other DEG external segmenters (`SAI3D`, `Open3DIS`, `SegmentAnything3D`) by:

- Accepting DEG external CLI inputs (`observations_path`, `scene_path`, optional flags)
- Producing `ObjectSegmentations` with `InstanceMaskObjectsDef`
- Printing exactly one parseable result line:
  - `objects_path: <path/to/objectsdef.pkl>`

## Current State

- Pixi environment exists in `dependencies/SAMPro3D/pyproject.toml`.
- Upstream SAMPro3D scripts are present:
  - `3d_prompt_proposal.py`
  - `main.py`
- No DEG bridge entrypoint yet (`segment.py` and integration glue are missing).

## Target Interface Contract (must match DEG)

The external runner (`src/deg/segmentationinitializers/external.py`) requires:

1. Process exits `0` on success.
2. Stdout contains a line starting with `objects_path: `.
3. The file at `objects_path` can be loaded via `ObjectSegmentations.load(...)`.

Expected output type:

- `ObjectSegmentations(object_segmentations=InstanceMaskObjectsDef(...))`
- `frame_ids`: ordered to match input observation frames
- `pixel_object_ids`: list of `H x W` integer masks
- `0` = background / no object

## Implementation Plan

### 1) Add SAMPro3D DEG bridge module

Create `dependencies/SAMPro3D/segmenter.py` with a public function:

- `initialize_scene(observations: Observations, scene: SceneSetup, intermediate_outputs_path: Optional[Path], ...flags) -> ObjectSegmentations`

Core steps inside `initialize_scene`:

1. Convert `ObservationFrame` to `psdframe.Frame` (same pattern as SAI3D/Open3DIS).
2. Reconstruct TSDF mesh from RGB-D frames.
3. Build workspace voxels from `scene.ground_gaussians` and crop mesh to workspace.
4. Write a temporary ScanNet-like dataset folder expected by upstream SAMPro3D.
5. Ensure SAM checkpoint is available.
6. Run SAMPro3D stage scripts (`3d_prompt_proposal.py`, then `main.py`) via subprocess.
7. Load predicted vertex labels from SAMPro3D output (`*_seg.npy`).
8. Convert vertex labels to per-frame pixel masks using raycast projection.
9. Apply DEG-aligned post-filtering:
   - keep only labels visible in at least 3 frames
   - detect and remove table instance
10. Build and return `ObjectSegmentations(InstanceMaskObjectsDef(...))`.

### 2) Add CLI entrypoint for external protocol

Create `dependencies/SAMPro3D/segment.py` similar to `dependencies/SAI3D/segment.py`:

- Tyro dataclass args with positional:
  - `observations_path`
  - `scene_path`
- Optional passthrough flags for SAMPro3D parameters, e.g.:
  - `device`, `model_type`, `sam_checkpoint`, `voxel_size`
  - `pred_iou_thres`, `stability_score_thres`, `box_nms_thres`, `keep_thres`
  - optional `post_floor` toggle (likely default `False` in DEG mode)
- Load `Observations` + `SceneSetup`, call `initialize_scene`, save to `objectsdef.pkl`.
- Print `objects_path: <...>` as final stdout line.

Note: keep verbose logging on stderr so stdout remains easy to parse.

### 3) Build ScanNet-style temporary export writer

Inside `segmenter.py`, implement a helper that writes this structure:

- `<work_root>/dataset/scannet/intrinsics.txt`
- `<work_root>/dataset/scannet/<scene_name>/color/{index}.jpg`
- `<work_root>/dataset/scannet/<scene_name>/depth/{index}.png` (uint16, millimeters)
- `<work_root>/dataset/scannet/<scene_name>/pose/{index}.txt` (`X_WV_opencv`)
- `<work_root>/dataset/scannet/<scene_name>/<scene_name>_vh_clean_2.ply`

Important details:

- Use sequential numeric frame filenames (consistent with SAMPro3D expectations).
- Convert color from float `[0,1]` to `uint8`.
- Convert depth meters to `uint16` mm.
- Write intrinsics in format expected by SAMPro3D (`4x4` matrix).

### 4) Reuse robust helpers from existing segmenters

To reduce risk and behavior drift, copy/adapt from SAI3D/Open3DIS where possible:

- TSDF mesh extraction and post-processing
- Workspace voxel computation and mesh cropping
- Segmentator-friendly mesh handling (if needed)
- Vertex-label to per-frame mask raycasting
- Table-id detection

Preferred reference files:

- `dependencies/SAI3D/segmenter.py`
- `dependencies/Open3DIS/segment.py`

### 5) Update SAMPro3D pixi task wiring

In `dependencies/SAMPro3D/pyproject.toml`:

- Set/confirm external task:
  - `segment_external = "python segment.py"`
- Keep upstream task(s) for direct usage:
  - `proposal = "python 3d_prompt_proposal.py"`
  - `segment = "python main.py"` (optional to keep)
- Ensure dependencies include bridge runtime libs (`tyro`, `initializerdefs`, `psdframe`, etc.).

### 6) Register in DEG external initializer list

Add shell launcher:

- `scripts/external_segmentation_initializers/sampro3d.sh`
  - `cd .../dependencies/SAMPro3D`
  - `pixi run --frozen segment_external "$@"`

Register in:

- `scripts/external_segmentation_initializers/__init__.py`

Add new segmentation type in:

- `src/deg/segmentationinitializers/runner.py`
  - include `"sampro3d"` in the `Literal[...]`

Optional UI exposure:

- Add a radio option in `scripts/deg_gui.py`.

## Recommended File-by-File Work Order

1. `dependencies/SAMPro3D/segmenter.py`
2. `dependencies/SAMPro3D/segment.py`
3. `dependencies/SAMPro3D/pyproject.toml`
4. `scripts/external_segmentation_initializers/sampro3d.sh`
5. `scripts/external_segmentation_initializers/__init__.py`
6. `src/deg/segmentationinitializers/runner.py`
7. (Optional) `scripts/deg_gui.py`

## Validation Checklist

### Local SAMPro3D package checks

- `pixi run --frozen segment_external -h`
- `pixi run --frozen python -c "import pointops; print('pointops ok')"`

### External protocol smoke test

Run via DEG external runner path and verify:

- process exits successfully
- stdout includes `objects_path: ...`
- returned file loads with `ObjectSegmentations.load(...)`
- mask count equals frame count
- each mask shape matches source frame
- mask dtype/int semantics are correct (`0` background)

### Integration run

From normal DEG workflow (`run_modelling.py` / GUI):

- select `segmentation.type = sampro3d`
- ensure pipeline continues through particle + semantic stages
- no contract errors from external runner

## Risks and Mitigations

- **High runtime / GPU memory pressure**
  - Mitigate with frame subsampling option and configurable SAM model/checkpoint.
- **Coordinate or convention mismatch (pose/intrinsics/depth scale)**
  - Validate with a tiny scene and visual debug outputs (projected masks).
- **Label convention mismatch (`-1` vs `0`)**
  - Normalize before creating `InstanceMaskObjectsDef`.
- **Non-determinism**
  - mirror deterministic seed setup from other segmenters.

## Out of Scope for First Pass

- Performance optimization and caching beyond basic output folders
- Advanced per-dataset variants beyond generated ScanNet-like export
- Semantic labeling integration (this is segmentation-only)

## Definition of Done

Integration is complete when all of the following are true:

1. `sampro3d` is selectable as a DEG segmentation initializer.
2. It runs through external initializer pipeline without manual patching.
3. It returns valid `ObjectSegmentations` in expected format.
4. End-to-end DEG initialization proceeds with those segmentations.
