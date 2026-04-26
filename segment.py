"""CLI entry point for SAMPro3D external segmenter.

Usage (invoked by pixi task):
    python segment.py <observations_path> <scene_path>

Outputs ``objects_path: <path>`` to stdout for the parent process to read.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import contextlib
import logging
import random
import sys
import time
import numpy as np
import torch
import tyro

from initializerdefs import Observations, SceneSetup
from segmenter import initialize_scene


DEFAULT_SEED = 42


@dataclass
class Args:
    observations_path: tyro.conf.Positional[Path]
    """Path to the pickled Observations."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled SceneSetup."""

    device: str = "cuda"
    """Device to run SAM on."""

    model_type: str = "vit_h"
    """SAM model type (vit_h, vit_l, vit_b)."""

    sam_checkpoint: Optional[Path] = None
    """Path to SAM checkpoint (auto-downloaded if omitted)."""

    voxel_size: float = 0.2
    """Voxel size for FPS prompt initialization."""

    pred_iou_thres: float = 0.7
    """Predicted IoU threshold for prompt filter."""

    stability_score_thres: float = 0.6
    """Stability score threshold for prompt filter."""

    box_nms_thres: float = 0.8
    """Box NMS threshold for prompt filter."""

    keep_thres: float = 0.4
    """Keep ratio threshold for prompt filter."""

    post_floor: bool = False
    """Whether to apply floor post-processing (default False for DEG)."""

    scene_ht_thres: float = 0.08
    """Height threshold for floor area proposal."""

    scene_inter_thres: float = 0.4
    """Intersection threshold for floor merging."""

    scene_dist_thres: float = 0.01
    """Distance threshold for RANSAC floor refinement."""

    intermediate_outputs_path: Optional[Path] = None
    """Optional directory to save intermediate outputs."""


def run() -> None:
    logger = logging.getLogger("sampro3d-segmenter")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    args = tyro.cli(Args)

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    with contextlib.redirect_stdout(sys.stderr):
        logger.info("--------------")
        logger.info("Starting SAMPro3D initialization")
        logger.info("params: %s", args)
        logger.info("Determinism enabled with seed=%d", DEFAULT_SEED)

        logger.info("Loading observations from %s ...", args.observations_path)
        dataset: Observations = Observations.load(args.observations_path)
        logger.info("Observations loaded.")

        logger.info("Loading scene setup from %s ...", args.scene_path)
        scene = SceneSetup.load(args.scene_path)
        logger.info("Scene loaded.")

        if dataset.id is None:
            logger.info("Dataset has no id, using transient id")
            dataset.id = f"transient_{time.strftime('%Y%m%d-%H%M%S')}"

        project_root = Path(__file__).parent
        output_dir = project_root / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset.id}"

        logger.info("Initializing scene ...")
        objects = initialize_scene(
            dataset,
            scene,
            intermediate_outputs_path=output_dir,
            device=args.device,
            model_type=args.model_type,
            sam_checkpoint=args.sam_checkpoint,
            voxel_size=args.voxel_size,
            pred_iou_thres=args.pred_iou_thres,
            stability_score_thres=args.stability_score_thres,
            box_nms_thres=args.box_nms_thres,
            keep_thres=args.keep_thres,
            post_floor=args.post_floor,
            scene_ht_thres=args.scene_ht_thres,
            scene_inter_thres=args.scene_inter_thres,
            scene_dist_thres=args.scene_dist_thres,
        )

        output_path = output_dir / "objectsdef.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        objects.save(output_path)

        logger.info("Objects saved to %s", output_path)

    # Output in format expected by ExternalSegmentationInitializer
    print(f"objects_path: {output_path}")


if __name__ == "__main__":
    run()
