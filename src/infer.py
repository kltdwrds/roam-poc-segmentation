"""
Inference: load a pretrained semantic segmenter, run it on a single indoor
scene, write three PLYs (raw / semantic-colored / permanent-only) for the
three-pane viewer.

Two model paths are supported:

  --model sparseconv   Day-1 path. SparseConvUnet (Open3D-ML, ScanNet-20),
                       runs on CPU/MPS. Default input is the bundled Redwood
                       Living Room fragment if no scene path is supplied.

  --model ptv3         Day-2 path. PTv3 (Pointcept, S3DIS-13), CUDA. Takes a
                       preprocessed S3DIS scene directory (with coord/color/
                       normal/segment.npy) via --scene-path, or picks a
                       fixed Area-5 scene if --scene-id is given.

Both paths write the same three PLYs (raw/semantic/permanent) so the viewer
is agnostic to which model produced them.

Usage:
    # Day 1
    python src/infer.py                                          # sparseconv on Redwood
    python src/infer.py --model sparseconv --scene-path X.ply

    # Day 2
    python src/infer.py --model ptv3 --scene-path /workspace/data/Area_5/office_1
    python src/infer.py --model ptv3 --scene-id Area_5/office_1
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from permanent_transient import (  # noqa: E402
    SCANNET_20_NAMES, SCANNET_20_PALETTE, SCANNET_20_BUCKET,
    S3DIS_13_NAMES, S3DIS_13_PALETTE, S3DIS_13_BUCKET,
    Bucket, permanent_mask,
)
from export_ply import write_colored_ply, subsample_to_budget  # noqa: E402


# ---------------------------------------------------------------------------
# Day-1 path: SparseConvUnet on ScanNet-20 (CPU/MPS via Open3D-ML)
# ---------------------------------------------------------------------------

def _sparseconv_load_model():
    import open3d as o3d
    from open3d._ml3d.torch.models.sparseconvnet import SparseConvUnet
    from open3d._ml3d.utils import Config
    cfg_path = Path(o3d.__file__).parent / "_ml3d" / "configs" / "sparseconvunet_scannet.yml"
    cfg = Config.load_from_file(str(cfg_path))
    model_cfg = {k: v for k, v in cfg.model.items() if k != "name"}
    model = SparseConvUnet(**model_cfg)
    ckpt = torch.load(str(ROOT / "checkpoints" / "sparseconvunet_scannet.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def _sparseconv_preprocess_keep_index(model, xyz, feat):
    """Mirrors SparseConvUnet.preprocess but also returns the surviving index
    array so predictions can be projected back onto the original point order."""
    cfg = model.cfg
    pts = xyz.astype(np.float32) / cfg.voxel_size
    m, M = pts.min(0), pts.max(0)
    grid = cfg.grid_size
    rng = np.random.default_rng(0)  # deterministic for inference
    offset = (
        -m
        + np.clip(grid - M + m - 0.001, 0, None) * rng.random(3)
        + np.clip(grid - M + m + 0.001, None, 0) * rng.random(3)
    )
    pts += offset
    keep = (pts.min(1) >= 0) & (pts.max(1) < grid)
    pts = pts[keep]
    feat = feat[keep]
    pts = (pts.astype(np.int32) + 0.5).astype(np.float32)
    return pts, feat, keep


def _sparseconv_run_inference(model, xyz, rgb_float):
    from open3d._ml3d.torch.dataloaders.concat_batcher import SparseConvUnetBatch
    pts, feat, keep = _sparseconv_preprocess_keep_index(model, xyz, rgb_float.astype(np.float32))
    fake_label = np.zeros(len(pts), dtype=np.int32)
    sample = {"data": {
        "point": torch.from_numpy(pts),
        "feat": torch.from_numpy(feat),
        "label": torch.from_numpy(fake_label),
    }}
    batch = SparseConvUnetBatch([sample])
    batch.to("cpu")
    with torch.no_grad():
        logits = model(batch)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_kept = probs.argmax(axis=1)
    pred = np.zeros(len(xyz), dtype=np.int64)
    pred[keep] = pred_kept + 1  # SparseConvUnet outputs 20 valid logits (0..19); shift to 1..20
    return pred


def _sparseconv_load_scene(path, fragment):
    import open3d as o3d
    if path is None:
        ds = o3d.data.LivingRoomPointClouds()
        path = ds.paths[fragment]
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_float = (np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors()
                 else np.full_like(xyz, 0.5, dtype=np.float32))
    return xyz, rgb_float, path


# ---------------------------------------------------------------------------
# Day-2 path: PTv3 on S3DIS-13 (CUDA via Pointcept)
# ---------------------------------------------------------------------------

def _ptv3_run(scene_dir, ckpt_path, device="cuda", finetuned=False):
    """Load PTv3 (raw or fine-tuned-with-PT-head), predict, return
    (xyz, rgb_u8, pred, gt) on the ORIGINAL pre-voxel ordering so we can
    write per-point PLYs.

    Two ckpt formats supported:
      - raw Pointcept format: top-level 'state_dict' with keys 'module.seg_head.*',
        'module.backbone.*'. Used by checkpoints/ptv3_s3dis.pth.
      - JointModel format (Day-2 fine-tune): top-level 'state_dict' with keys
        'seg_model.seg_head.*', 'seg_model.backbone.*', 'pt_head.*'. Used by
        checkpoints/finetune/ptv3_pt_epoch*.pth.

    `finetuned=True` selects the second loader and uses the PT head for the
    permanent/transient prediction. Sem PLY still uses S3DIS-13 semantic
    palette either way."""
    from ptv3_infer import (
        build_ptv3_s3dis, load_ptv3_ckpt, load_s3dis_scene_raw,
        run_ptv3_inference, prepare_val_input,
    )
    scene = load_s3dis_scene_raw(scene_dir)
    xyz = scene["coord"]
    rgb_u8 = scene["color"].clip(0, 255).astype(np.uint8)

    if not finetuned:
        model = build_ptv3_s3dis().to(device)
        load_ptv3_ckpt(model, ckpt_path, strict=True)
        pred, gt, _ = run_ptv3_inference(model, scene, device=device)
        return xyz, rgb_u8, pred, gt, str(scene_dir)

    # Fine-tuned JointModel path
    from finetune import JointModel
    seg = build_ptv3_s3dis().to(device)
    joint = JointModel(seg).to(device)
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    joint.load_state_dict(ckpt["state_dict"], strict=True)
    joint.eval()

    # Voxelize + forward (fp32 — bf16 caused spconv autotune failures on
    # whole-scene inference; see env/SETUP.md eval section).
    data = prepare_val_input(scene)
    inp = {k: v.to(device) if hasattr(v, "to") else v for k, v in data.items()}
    with torch.no_grad():
        sem_logits, _pt_logits = joint(inp)
    pred_v = sem_logits.argmax(-1).cpu().numpy().astype(np.int64)

    inverse = data["inverse"]
    if hasattr(inverse, "cpu"):
        inverse = inverse.cpu().numpy()
    inverse = inverse.astype(np.int64)
    pred = pred_v[inverse]
    gt = scene["segment"].astype(np.int64)
    return xyz, rgb_u8, pred, gt, str(scene_dir)


# ---------------------------------------------------------------------------
# Summary + PLY export (shared across both model paths)
# ---------------------------------------------------------------------------

def summarize(pred: np.ndarray, names: dict, bucket_map: dict, label_space: str) -> None:
    total = len(pred)
    print(f"\n[{label_space}] Class distribution over {total:,} points:")
    print(f"{'id':>3}  {'name':<16}  {'bucket':<10}  {'count':>10}  {'pct':>6}")
    for cid, name in names.items():
        n = int((pred == cid).sum())
        if n == 0:
            continue
        pct = 100.0 * n / total
        bucket = bucket_map[cid].value
        print(f"{cid:>3}  {name:<16}  {bucket:<10}  {n:>10,}  {pct:>5.1f}%")
    perm = int(permanent_mask(pred).sum())
    print(f"\nPERMANENT: {perm:,} ({100.0*perm/total:.1f}%)   "
          f"TRANSIENT: {total-perm:,} ({100.0*(total-perm)/total:.1f}%)")


def export_three_plys(out_dir: Path, xyz, rgb_u8, pred, palette: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    pal = np.zeros((max(palette) + 1, 3), dtype=np.uint8)
    for cid, color in palette.items():
        pal[cid] = color
    sem_rgb = pal[pred]
    perm_mask = permanent_mask(pred)

    raw_xyz, raw_rgb = subsample_to_budget(xyz, rgb_u8)[:2]
    write_colored_ply(str(out_dir / "scene_raw.ply"), raw_xyz, raw_rgb)
    write_colored_ply(str(out_dir / "scene_semantic.ply"),
                      *subsample_to_budget(xyz, sem_rgb)[:2])

    perm_xyz = xyz[perm_mask]
    perm_rgb = rgb_u8[perm_mask]
    if len(perm_xyz):
        write_colored_ply(str(out_dir / "scene_permanent.ply"),
                          *subsample_to_budget(perm_xyz, perm_rgb)[:2])
    else:
        print("  [warn] no permanent points predicted; skipping scene_permanent.ply")

    for f in ("scene_raw.ply", "scene_semantic.ply", "scene_permanent.ply"):
        p = out_dir / f
        if p.exists():
            print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("sparseconv", "ptv3"), default="sparseconv",
                    help="sparseconv = Day-1 ScanNet path (CPU). ptv3 = Day-2 S3DIS path (CUDA).")
    ap.add_argument("--scene-path", default=None,
                    help="sparseconv: .ply file. ptv3: S3DIS scene directory (has coord/color/normal/segment.npy).")
    ap.add_argument("--scene-id", default=None,
                    help="ptv3 only: shorthand like 'Area_5/office_1' resolved against --s3dis-root.")
    ap.add_argument("--s3dis-root", default="/workspace/data",
                    help="ptv3 only: root of preprocessed S3DIS (default /workspace/data)")
    ap.add_argument("--ckpt", default=None,
                    help="ptv3 only: path to PTv3 checkpoint (default checkpoints/ptv3_s3dis.pth)")
    ap.add_argument("--finetuned", action="store_true",
                    help="ptv3 only: load Day-2 fine-tune JointModel ckpt format "
                         "(checkpoints/finetune/ptv3_pt_epoch*.pth). Uses fine-tuned weights.")
    ap.add_argument("--fragment", type=int, default=12,
                    help="sparseconv only: Redwood living-room fragment index (default: 12)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default outputs/ for sparseconv, outputs/day2_pretrained/<scene>/ for ptv3)")
    args = ap.parse_args()

    if args.model == "sparseconv":
        out_dir = Path(args.out_dir or (ROOT / "outputs"))
        print("Loading scene...")
        xyz, rgb_float, src = _sparseconv_load_scene(args.scene_path, args.fragment)
        print(f"  source: {src}")
        print(f"  points: {len(xyz):,}")
        print("Loading pretrained SparseConvUnet (ScanNet-20)...")
        model = _sparseconv_load_model()
        print("Running inference (CPU)...")
        pred = _sparseconv_run_inference(model, xyz, rgb_float)
        summarize(pred, SCANNET_20_NAMES, SCANNET_20_BUCKET, "ScanNet-20")
        rgb_u8 = (rgb_float * 255.0).astype(np.uint8)
        print("\nWriting PLYs...")
        export_three_plys(out_dir, xyz, rgb_u8, pred, SCANNET_20_PALETTE)
        print("\nDone.")
        return

    # PTv3 path
    if args.scene_id:
        scene_dir = Path(args.s3dis_root) / args.scene_id
    elif args.scene_path:
        scene_dir = Path(args.scene_path)
    else:
        ap.error("ptv3: --scene-id or --scene-path required")
    if not scene_dir.is_dir():
        ap.error(f"scene dir not found: {scene_dir}")

    ckpt_path = Path(args.ckpt or (ROOT / "checkpoints" / "ptv3_s3dis.pth"))
    if not ckpt_path.is_file():
        ap.error(f"PTv3 ckpt not found: {ckpt_path}")

    scene_tag = f"{scene_dir.parent.name}-{scene_dir.name}"
    default_subdir = "day2_final" if args.finetuned else "day2_pretrained"
    out_dir = Path(args.out_dir or (ROOT / "outputs" / default_subdir / scene_tag))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading PTv3 S3DIS (rpe ckpt) on {device}...")
    print(f"  scene: {scene_dir}")
    print(f"  ckpt:  {ckpt_path} ({ckpt_path.stat().st_size/1e6:.0f} MB)")
    xyz, rgb_u8, pred, gt, src = _ptv3_run(scene_dir, ckpt_path, device=device, finetuned=args.finetuned)
    print(f"  points: {len(xyz):,}")
    summarize(pred, S3DIS_13_NAMES, S3DIS_13_BUCKET, "S3DIS-13")
    if gt is not None:
        # Per-class accuracy & overall as a quick check
        valid = gt >= 0
        acc = float((pred[valid] == gt[valid]).mean())
        print(f"\nOverall per-point accuracy vs GT: {acc*100:.1f}%")
    print("\nWriting PLYs...")
    export_three_plys(out_dir, xyz, rgb_u8, pred, S3DIS_13_PALETTE)
    print("\nDone.")


if __name__ == "__main__":
    main()
