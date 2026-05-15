"""
Step 7: evaluate fine-tuned PTv3 checkpoints on S3DIS Area-5.

Loads a JointModel (PTv3 + PT head) checkpoint, runs inference scene-by-scene
across all Area-5 scenes (with Pointcept's val transform — same one used
during training to compute per-step metrics), accumulates intersection/union
per S3DIS-13 class, and reports:

  - Per-class IoU + mean IoU
  - Overall per-point accuracy
  - **Binary** precision / recall / F1 for PERMANENT vs TRANSIENT (the
    headline number for the demo). Bucketing uses S3DIS_13_BUCKET.

Two prediction sources are evaluated:
  - "sem_arg": argmax over the 13 semantic logits, then S3DIS_13_BUCKET lookup
  - "pt_arg":  argmax over the PT head's 2 logits

We expect "pt_arg" to outperform "sem_arg" slightly if the auxiliary head
has learned anything beyond hard-coded bucketing — that delta is the
demo-relevant signal.

Usage:
    PYTHONPATH=/workspace/Pointcept python src/eval.py \\
        --ckpt checkpoints/finetune/ptv3_pt_epoch02.pth \\
        --out eval/results_epoch02.json

For the multi-ckpt sweep, just call repeatedly with different --ckpt/--out.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from permanent_transient import (  # noqa: E402
    S3DIS_13_NAMES, S3DIS_13_BUCKET, Bucket, permanent_mask,
)
from ptv3_infer import build_ptv3_s3dis, S3DIS_VAL_TRANSFORM_CFG, load_s3dis_scene_raw  # noqa: E402
from finetune import JointModel, build_pt_lut, NUM_SEM_CLASSES  # noqa: E402

from pointcept.datasets.transform import Compose  # noqa: E402


def load_joint_ckpt(joint: JointModel, ckpt_path: str | Path) -> dict:
    """Load a fine-tune ckpt produced by finetune.py:save_weights_only."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    joint.load_state_dict(sd, strict=True)
    return ckpt


@torch.no_grad()
def predict_scene(joint: JointModel, scene_dir: Path, device: str, amp: bool):
    """Single-pass val-mode prediction for one Area-5 scene.

    Returns (gt_orig, sem_pred_orig, pt_pred_orig) all shape (N_original,)."""
    scene = load_s3dis_scene_raw(scene_dir)
    compose = Compose(S3DIS_VAL_TRANSFORM_CFG)
    data = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in scene.items()}
    data = compose(data)
    n = data["coord"].shape[0]
    data["offset"] = torch.tensor([n], dtype=torch.long)
    inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    joint.eval()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        sem_logits, pt_logits = joint(inp)
    sem_v = sem_logits.argmax(-1).cpu().numpy().astype(np.int64)
    pt_v = pt_logits.argmax(-1).cpu().numpy().astype(np.int64)

    inverse = data["inverse"]
    if isinstance(inverse, torch.Tensor):
        inverse = inverse.cpu().numpy()
    inverse = inverse.astype(np.int64)

    return scene["segment"].astype(np.int64), sem_v[inverse], pt_v[inverse]


def accumulate_class_stats(stats: dict, gt: np.ndarray, pred: np.ndarray, n_classes: int):
    """Accumulate intersection / union / target counts per class."""
    valid = gt >= 0
    g = gt[valid]; p = pred[valid]
    for c in range(n_classes):
        tp = int(((p == c) & (g == c)).sum())
        fp = int(((p == c) & (g != c)).sum())
        fn = int(((p != c) & (g == c)).sum())
        stats["tp"][c] += tp
        stats["fp"][c] += fp
        stats["fn"][c] += fn
        stats["target"][c] += int((g == c).sum())


def accumulate_binary_stats(stats: dict, gt_bin: np.ndarray, pred_bin: np.ndarray):
    """Binary perm/transient counts; convention: PERMANENT = 1, TRANSIENT = 0
    (i.e., we report F1 for the PERMANENT class — that's the headline number
    because the demo cares about "what to keep in the persistent index")."""
    p1 = (pred_bin == 1) & (gt_bin >= 0)
    t1 = (gt_bin == 1)
    stats["tp"] += int((p1 & t1).sum())
    stats["fp"] += int((p1 & ~t1).sum())
    stats["fn"] += int((~p1 & t1 & (gt_bin >= 0)).sum())
    stats["tn"] += int((~p1 & ~t1 & (gt_bin >= 0)).sum())


def summarize_classes(stats: dict, names: dict, n_classes: int):
    ious = []
    rows = []
    for c in range(n_classes):
        tp = stats["tp"][c]; fp = stats["fp"][c]; fn = stats["fn"][c]
        union = tp + fp + fn
        iou = tp / union if union else float("nan")
        if union:
            ious.append(iou)
        rows.append(dict(
            class_id=c, name=names[c], tp=tp, fp=fp, fn=fn, target=stats["target"][c],
            iou=iou,
        ))
    m_iou = float(np.nanmean(ious)) if ious else float("nan")
    total_correct = sum(stats["tp"])
    total_points = sum(stats["target"])
    acc = total_correct / total_points if total_points else float("nan")
    return rows, m_iou, acc


def summarize_binary(stats: dict):
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--s3dis-root", default="/workspace/data")
    ap.add_argument("--area", default="Area_5")
    ap.add_argument("--out", default=None, help="output results json path")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="eval only first N scenes (debug)")
    args = ap.parse_args()

    device = "cuda"
    out_path = Path(args.out or (ROOT / "eval" / f"results_{Path(args.ckpt).stem}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seg = build_ptv3_s3dis(num_classes=NUM_SEM_CLASSES)
    joint = JointModel(seg).to(device)
    ckpt_meta = load_joint_ckpt(joint, args.ckpt)
    print(f"Loaded ckpt: {args.ckpt}  epoch={ckpt_meta.get('epoch')}")

    pt_lut_np = build_pt_lut().numpy()  # S3DIS class id -> 0(perm)/1(trans)
    # NOTE: In the LUT, 0=PERMANENT, 1=TRANSIENT. For the headline binary F1,
    # we want F1 of the PERMANENT class. Convert to "perm=1" convention here.
    perm_one = (1 - pt_lut_np).astype(np.int64)  # 0=transient, 1=permanent

    scene_dirs = sorted(p for p in (Path(args.s3dis_root) / args.area).iterdir() if p.is_dir())
    if args.limit:
        scene_dirs = scene_dirs[:args.limit]
    print(f"Evaluating {len(scene_dirs)} scenes from {args.area}")

    n_classes = NUM_SEM_CLASSES
    cls_stats = dict(tp=[0]*n_classes, fp=[0]*n_classes, fn=[0]*n_classes, target=[0]*n_classes)
    bin_stats_sem = dict(tp=0, fp=0, fn=0, tn=0)  # sem->bucket lookup
    bin_stats_pt = dict(tp=0, fp=0, fn=0, tn=0)   # PT-head direct
    confusion_2x2_sem = np.zeros((2, 2), dtype=np.int64)  # rows=gt(0=trans,1=perm), cols=pred
    confusion_2x2_pt = np.zeros((2, 2), dtype=np.int64)

    t0 = time.time()
    for i, scene_dir in enumerate(scene_dirs):
        gt, sem_pred, pt_pred_raw = predict_scene(joint, scene_dir, device, amp=not args.no_amp)
        # GT binary
        valid = gt >= 0
        gt_perm = perm_one[gt.clip(0)] * valid  # 0/1, with valid mask

        # Sem -> bucket
        sem_perm = perm_one[sem_pred] * valid

        # PT head: raw output is 0=permanent, 1=transient (LUT convention).
        # Convert to perm_one: 0->1, 1->0.
        pt_perm = (1 - pt_pred_raw) * valid

        accumulate_class_stats(cls_stats, gt, sem_pred, n_classes)
        accumulate_binary_stats(bin_stats_sem, gt_perm.astype(np.int64), sem_perm.astype(np.int64))
        accumulate_binary_stats(bin_stats_pt, gt_perm.astype(np.int64), pt_perm.astype(np.int64))
        # Confusion matrices
        for gp, pp, cm in [(gt_perm, sem_perm, confusion_2x2_sem),
                           (gt_perm, pt_perm, confusion_2x2_pt)]:
            mask = valid
            cm[0, 0] += int(((gp == 0) & (pp == 0) & mask).sum())
            cm[0, 1] += int(((gp == 0) & (pp == 1) & mask).sum())
            cm[1, 0] += int(((gp == 1) & (pp == 0) & mask).sum())
            cm[1, 1] += int(((gp == 1) & (pp == 1) & mask).sum())

        print(f"  [{i+1}/{len(scene_dirs)}] {scene_dir.name:<25} N={gt.size:>9,}  "
              f"sem_acc={(sem_pred[valid]==gt[valid]).mean()*100:5.1f}%  "
              f"perm_pred(sem)={sem_perm[valid].mean()*100:4.1f}%  "
              f"perm_pred(pt)={pt_perm[valid].mean()*100:4.1f}%", flush=True)

    print(f"\nEvaluated in {(time.time()-t0)/60:.1f} min")

    rows, m_iou, acc = summarize_classes(cls_stats, S3DIS_13_NAMES, n_classes)
    bin_sem = summarize_binary(bin_stats_sem)
    bin_pt = summarize_binary(bin_stats_pt)

    print(f"\n=== Per-class IoU (S3DIS-13) ===")
    print(f"{'class':<10} {'IoU':>7} {'tp':>10} {'fp':>10} {'fn':>10}")
    for r in rows:
        print(f"  {r['name']:<10} {r['iou']*100:6.1f}% {r['tp']:>10d} {r['fp']:>10d} {r['fn']:>10d}")
    print(f"\nmIoU: {m_iou*100:.2f}%   overall point acc: {acc*100:.2f}%")

    print(f"\n=== Binary PERMANENT vs TRANSIENT (headline) ===")
    print(f"  sem->bucket :  P={bin_sem['precision']*100:.1f}%  R={bin_sem['recall']*100:.1f}%  "
          f"F1={bin_sem['f1']*100:.1f}%")
    print(f"  PT head     :  P={bin_pt['precision']*100:.1f}%  R={bin_pt['recall']*100:.1f}%  "
          f"F1={bin_pt['f1']*100:.1f}%")

    payload = dict(
        ckpt=str(args.ckpt),
        epoch=ckpt_meta.get("epoch"),
        area=args.area,
        n_scenes=len(scene_dirs),
        per_class=rows,
        mIoU=m_iou,
        overall_acc=acc,
        binary_sem=bin_sem,
        binary_pt=bin_pt,
        confusion_sem=confusion_2x2_sem.tolist(),
        confusion_pt=confusion_2x2_pt.tolist(),
        elapsed_min=(time.time() - t0) / 60.0,
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
