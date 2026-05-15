"""
Day-2 fine-tune: PTv3 (Pointcept) on S3DIS with a joint permanent/transient
auxiliary head.

Loads the (broken-but-strict-loading) pretrained PTv3 S3DIS rpe checkpoint
as initialization, attaches a PTHead on top of the 13-class semantic
logits, and trains both jointly:

    loss = ce(sem_logits, segment) + lambda * ce(pt_logits, pt_target)

where pt_target = LUT[segment] (7 permanent S3DIS classes → 0, 6 transient → 1).

Train split: S3DIS Areas 1/2/3/4/6  (val: Area 5 — held out).

Design notes:
  - We use Pointcept's own `build_dataset` + Compose pipeline (copying the
    train/val transform lists from configs/s3dis/semseg-pt-v3m1-1-rpe.py)
    instead of reimplementing — Day-2's first lesson was "use the framework's
    data pipeline, don't reimplement". A custom collate_fn would also work
    but Pointcept's `collate_fn` handles offsets correctly.
  - Disk is tight (3 GB free). Save **model weights only** per epoch, not
    optimizer state. One full PTv3 ckpt is ~180 MB; 3 epochs × 180 MB fits.
  - Print every 50 steps: total / sem / pt loss + pt-head grad norm + pt-head
    weight norm. Day-1 carry-over: surfaces dead-gradient bugs early.
  - If the pretrained init poisons training (loss flat or exploding in
    first ~100 steps), pass --reinit-head to re-init the seg_head Linear
    layer and try again without re-downloading the ckpt.

Usage:
    PYTHONPATH=/workspace/Pointcept \\
    python src/finetune.py \\
        --s3dis-root /workspace/data \\
        --pretrained checkpoints/ptv3_s3dis.pth \\
        --epochs 3 \\
        --out-dir checkpoints/finetune \\
        --log-dir runs/finetune
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from permanent_transient import S3DIS_13_BUCKET, Bucket  # noqa: E402
from ptv3_infer import (  # noqa: E402
    PTV3_S3DIS_BACKBONE, build_ptv3_s3dis, load_ptv3_ckpt,
)

# Pointcept (via PYTHONPATH=/workspace/Pointcept)
from pointcept.datasets import build_dataset, point_collate_fn  # noqa: E402


# S3DIS-13 label space
NUM_SEM_CLASSES = 13


# ---------------------------------------------------------------------------
# Data pipeline — copied verbatim from configs/s3dis/semseg-pt-v3m1-1-rpe.py
# ---------------------------------------------------------------------------
# Trimmed `SphereCrop sample_rate=0.6` → keep `point_max=204800` only, so
# crops are deterministic-size 204k voxels; cuts variance in per-step time.

S3DIS_TRAIN_TRANSFORM_CFG = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="GridSample",
         grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="SphereCrop", point_max=204800, mode="random"),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect",
         keys=("coord", "grid_coord", "segment"),
         feat_keys=("color", "normal")),
]


def build_pt_lut() -> torch.Tensor:
    """LUT mapping S3DIS class id (0..12) to PT bucket (0=permanent, 1=transient)."""
    lut = torch.zeros(NUM_SEM_CLASSES, dtype=torch.long)
    for cid, bucket in S3DIS_13_BUCKET.items():
        lut[cid] = 0 if bucket == Bucket.PERMANENT else 1
    return lut


class JointModel(nn.Module):
    """Wraps a DefaultSegmentorV2(PTv3) and adds a 2-class PT head on top of
    the 13-class semantic logits.

    Forward returns (sem_logits, pt_logits). We bypass DefaultSegmentorV2's
    built-in `criteria` (a 13-class CE) and recompute losses ourselves so we
    can weight sem vs pt independently and log per-component loss. Concretely
    that means: call `backbone` + `seg_head` directly, skip the wrapper's
    forward (which in training mode swallows seg_logits into a loss dict).

    `seg_model` is a `DefaultSegmentorV2`. We reach into `.backbone` and
    `.seg_head` directly. The wrapper is still kept on `self.seg_model` so
    the checkpoint key names (`seg_model.backbone.*`, `seg_model.seg_head.*`)
    match Day-1's save format.
    """

    def __init__(self, seg_model: nn.Module):
        super().__init__()
        self.seg_model = seg_model
        self.pt_head = nn.Linear(NUM_SEM_CLASSES, 2)

    def forward(self, input_dict):
        from pointcept.models.utils.structure import Point
        point = Point(input_dict)
        point = self.seg_model.backbone(point)
        # PTv3's backbone returns a Point with feat in .feat after the
        # pooling-tree unwind. Mirror DefaultSegmentorV2's unwind logic:
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        sem_logits = self.seg_model.seg_head(feat)
        pt_logits = self.pt_head(sem_logits)
        return sem_logits, pt_logits


def reinit_seg_head(model: nn.Module):
    """Re-initialize the final segmentation Linear layer (debugging hook for
    when the pretrained init poisons fine-tuning)."""
    sh = model.seg_model.seg_head
    if isinstance(sh, nn.Linear):
        nn.init.trunc_normal_(sh.weight, std=0.02)
        if sh.bias is not None:
            nn.init.zeros_(sh.bias)
        print("[reinit] seg_head Linear re-initialized")


def make_dataloaders(s3dis_root: str, batch_size: int, num_workers: int):
    train_cfg = dict(
        type="S3DISDataset",
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=s3dis_root,
        transform=S3DIS_TRAIN_TRANSFORM_CFG,
        test_mode=False,
        loop=1,
    )
    train_ds = build_dataset(train_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=point_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader


def save_weights_only(path: Path, joint: JointModel, epoch: int, extra: dict):
    """Save model weights only (no optimizer state) to keep ckpt small.
    `extra` is a flat dict of small metadata items (e.g. step counts, lrs).
    """
    sd = {k: v.cpu() for k, v in joint.state_dict().items()}
    torch.save({"state_dict": sd, "epoch": epoch, **extra}, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3dis-root", default="/workspace/data")
    ap.add_argument("--pretrained", default=str(ROOT / "checkpoints" / "ptv3_s3dis.pth"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="bs per step. Each item = ~204k-voxel sphere crop. "
                         "PTv3 patch_size=128 fits bs=4 comfortably on A100 80GB.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4,
                    help="backbone lr; head gets 10x")
    ap.add_argument("--head-lr-mul", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--pt-weight", type=float, default=0.5,
                    help="lambda for the perm/transient auxiliary loss")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints" / "finetune"))
    ap.add_argument("--log-dir", default=str(ROOT / "runs" / "finetune"))
    ap.add_argument("--reinit-head", action="store_true",
                    help="re-init seg_head Linear after loading pretrained "
                         "(use if pretrained init poisons training)")
    ap.add_argument("--amp", action="store_true",
                    help="enable mixed precision (bf16). Big speedup on A100.")
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir / run_name))

    print(f"Device: {device}")
    print(f"Train areas: 1,2,3,4,6; held-out val: 5")
    print(f"Output ckpt dir: {out_dir}")
    print(f"Tensorboard run: {log_dir/run_name}")

    # --- model ---
    print("Building PTv3 + PT head ...")
    seg = build_ptv3_s3dis(num_classes=NUM_SEM_CLASSES)
    load_ptv3_ckpt(seg, args.pretrained, strict=True)
    joint = JointModel(seg).to(device)
    if args.reinit_head:
        reinit_seg_head(joint)

    pt_lut = build_pt_lut().to(device)

    # --- data ---
    print("Building train dataloader (this may take ~30s while it lists scenes) ...")
    train_loader = make_dataloaders(args.s3dis_root, args.batch_size, args.num_workers)
    print(f"  train scenes: {len(train_loader.dataset)}, steps/epoch: {len(train_loader)}")

    # --- optim: split backbone vs head LRs ---
    backbone_params, head_params = [], []
    for n, p in joint.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("pt_head") or n.startswith("seg_model.seg_head"):
            head_params.append(p)
        else:
            backbone_params.append(p)
    optim = torch.optim.AdamW(
        [
            dict(params=backbone_params, lr=args.lr),
            dict(params=head_params, lr=args.lr * args.head_lr_mul),
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=[args.lr, args.lr * args.head_lr_mul],
        total_steps=args.epochs * len(train_loader),
        pct_start=0.05, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=1000.0,
    )

    sem_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    pt_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        joint.train()
        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to GPU
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)
            seg_gt = batch["segment"].long()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                sem_logits, pt_logits = joint(batch)
                # pt_target: map valid seg labels through LUT; ignore -1
                valid = seg_gt >= 0
                pt_target = torch.full_like(seg_gt, -1)
                pt_target[valid] = pt_lut[seg_gt[valid].clamp_min(0)]
                sem_l = sem_loss_fn(sem_logits, seg_gt)
                pt_l = pt_loss_fn(pt_logits, pt_target)
                loss = sem_l + args.pt_weight * pt_l

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(joint.parameters(), args.clip_grad)
            scaler.step(optim)
            scaler.update()
            sched.step()

            if global_step % args.print_every == 0:
                # Gradient-health signals on the PT head (Day-1 pattern)
                head_w = joint.pt_head.weight
                head_grad_norm = (head_w.grad.detach().norm().item()
                                  if head_w.grad is not None else 0.0)
                head_w_norm = head_w.detach().norm().item()
                elapsed = time.time() - t0
                lr_bb = sched.get_last_lr()[0]
                lr_hd = sched.get_last_lr()[1]
                with torch.no_grad():
                    sem_pred = sem_logits.argmax(-1)
                    sem_acc = (sem_pred[valid] == seg_gt[valid]).float().mean().item()
                    pt_pred = pt_logits.argmax(-1)
                    pt_acc = (pt_pred[valid] == pt_target[valid]).float().mean().item()
                print(
                    f"e{epoch} s{global_step:5d} ({batch_idx+1}/{len(train_loader)}) "
                    f"loss={loss.item():.3f} (sem={sem_l.item():.3f} pt={pt_l.item():.3f}) "
                    f"sem_acc={sem_acc*100:.1f}% pt_acc={pt_acc*100:.1f}% "
                    f"head_grad={head_grad_norm:.2e} head_w={head_w_norm:.2f} "
                    f"lr_bb={lr_bb:.1e} lr_hd={lr_hd:.1e} "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/semantic", sem_l.item(), global_step)
                writer.add_scalar("loss/permanent_transient", pt_l.item(), global_step)
                writer.add_scalar("acc/sem_step", sem_acc, global_step)
                writer.add_scalar("acc/pt_step", pt_acc, global_step)
                writer.add_scalar("grad/pt_head_norm", head_grad_norm, global_step)
                writer.add_scalar("norm/pt_head_w", head_w_norm, global_step)
                writer.add_scalar("lr/backbone", lr_bb, global_step)
                writer.add_scalar("lr/head", lr_hd, global_step)
            global_step += 1

        # End of epoch — save lean ckpt
        ckpt_path = out_dir / f"ptv3_pt_epoch{epoch:02d}.pth"
        save_weights_only(ckpt_path, joint, epoch,
                          extra={"global_step": global_step, "args": vars(args)})
        print(f"  saved {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.0f} MB)")

    writer.close()
    print(f"Done. Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
