"""
PTv3 inference helpers (Day 2).

Loads Pointcept's pretrained PTv3 on S3DIS Area-5 and runs per-point
inference on a single S3DIS scene directory containing coord/color/normal
/segment npy files.

We reuse Pointcept's own `Compose` of transforms (matched to the official
S3DIS val config) so voxelization / centering / color normalization are
bit-for-bit identical to how the model was trained. Hand-rolling this
gave 32.8% per-point accuracy on a scene the model should score ~75% on
— moral: use the framework's data pipeline, don't reimplement it.

Kept separate from Day-1 `infer.py`'s SparseConvUnet path because:
  - different dependencies (Pointcept vs Open3D-ML)
  - different label space (S3DIS-13 vs ScanNet-20)
  - PTv3 needs voxel/grid coords + an "offset" tensor not "batch"
"""

from __future__ import annotations
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.models import build_model
from pointcept.datasets.transform import Compose


# Backbone config for the `semseg-pt-v3m1-0-rpe.py` checkpoint we trained
# against. patch_size MUST be 128 to strict-load that ckpt; enable_rpe=True,
# enable_flash=False (RPE and flash-attn are mutually exclusive in PTv3).
PTV3_S3DIS_BACKBONE = dict(
    type="PT-v3m1",
    in_channels=6,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(128, 128, 128, 128, 128),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(128, 128, 128, 128),
    mlp_ratio=4, qkv_bias=True, qk_scale=None,
    attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
    shuffle_orders=True, pre_norm=True,
    enable_rpe=True, enable_flash=False,
    upcast_attention=True, upcast_softmax=True,
    enc_mode=False,
    pdnorm_bn=False, pdnorm_ln=False, pdnorm_decouple=True,
    pdnorm_adaptive=False, pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
)


# Exactly the val transform list from configs/s3dis/semseg-pt-v3m1-0-rpe.py
S3DIS_VAL_TRANSFORM_CFG = [
    dict(type="CenterShift", apply_z=True),
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
        feat_keys=("color", "normal"),
    ),
]


def build_ptv3_s3dis(num_classes: int = 13) -> torch.nn.Module:
    """Build a DefaultSegmentorV2(PTv3) wrapper for S3DIS-13."""
    return build_model(dict(
        type="DefaultSegmentorV2",
        num_classes=num_classes,
        backbone_out_channels=64,
        backbone=PTV3_S3DIS_BACKBONE,
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
    ))


def load_ptv3_ckpt(model: torch.nn.Module, ckpt_path: str | os.PathLike, strict: bool = True) -> None:
    """Strict-load the pretrained Pointcept checkpoint into `model`.

    Released checkpoint is DDP-wrapped (keys prefixed with 'module.').
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=strict)


def load_s3dis_scene_raw(scene_dir: str | os.PathLike) -> dict:
    """Load a preprocessed S3DIS scene from its npy directory, untransformed."""
    scene_dir = Path(scene_dir)
    return dict(
        coord=np.load(scene_dir / "coord.npy").astype(np.float32),
        color=np.load(scene_dir / "color.npy").astype(np.float32),
        normal=np.load(scene_dir / "normal.npy").astype(np.float32),
        segment=np.load(scene_dir / "segment.npy").reshape(-1).astype(np.int64),
    )


def prepare_val_input(scene: dict) -> dict:
    """Apply Pointcept's S3DIS val transform list. Returns a torch input dict
    (single-scene batch) ready to feed into the model.

    NOTE: `Compose` mutates the dict in place; pass a shallow copy of the
    scene to avoid surprising the caller.
    """
    compose = Compose(S3DIS_VAL_TRANSFORM_CFG)
    data_dict = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in scene.items()}
    data_dict = compose(data_dict)
    # Single-scene "batch": offset = [N] (cumulative point counts).
    n = data_dict["coord"].shape[0]
    data_dict["offset"] = torch.tensor([n], dtype=torch.long)
    return data_dict


@torch.no_grad()
def run_ptv3_inference(
    model: torch.nn.Module,
    scene: dict,
    device: str = "cuda",
    return_voxel_pred: bool = False,
):
    """Run PTv3 on a loaded S3DIS scene dict.

    Returns:
      pred_per_point  — np.int64 array shape (N_original,), values 0..12
      gt_per_point    — np.int64 (N_original,) original-order GT from scene['segment']
      extra           — dict with voxel-level pred + inverse if requested
    """
    data = prepare_val_input(scene)
    # Move tensors to device
    model_in = {}
    for k, v in data.items():
        model_in[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    model.eval()
    out = model(model_in)
    logits = out["seg_logits"] if isinstance(out, dict) else out  # (Nv, 13)
    probs = F.softmax(logits, dim=-1)
    pred_v = probs.argmax(dim=-1).cpu().numpy().astype(np.int64)

    # Project back to original point ordering via the 'inverse' index built by
    # GridSample (val transform sets return_inverse=True).
    inverse = data["inverse"]
    if isinstance(inverse, torch.Tensor):
        inverse = inverse.cpu().numpy()
    inverse = inverse.astype(np.int64)
    pred_per_point = pred_v[inverse]

    gt = scene["segment"].astype(np.int64)  # original-order GT
    extra = {}
    if return_voxel_pred:
        extra["voxel_pred"] = pred_v
        extra["voxel_probs"] = probs.cpu().numpy()
        extra["inverse"] = inverse
    return pred_per_point, gt, extra
