"""
Day-2 fine-tune scaffold.

Loads the pretrained SparseConvUnet ScanNet checkpoint and attaches an
auxiliary 2-class permanent/transient head on top of the penultimate
features. Both heads are trained jointly:

  total_loss = ce(semantic_logits, y_semantic) + lambda * ce(pt_logits, y_pt)

On Day 2 (NVIDIA GPU) this will train on the full ScanNet train split. Day 1
we just need `python src/finetune.py --quick` to complete end-to-end on CPU
without crashing — that proves the wiring (data → forward → loss → backward
→ optimizer step → checkpoint save → tensorboard write) is sound before
spending GPU money.

Quick mode uses a small bag of random synthetic "rooms" so it can run with
no dataset present. Real mode requires `--scannet-root` and a ScanNet
training split prepared in the Pointcept layout. Real-mode loader is
stubbed (raises NotImplementedError with a clear message) — wiring it in
is the first thing to do on Day 2.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import open3d as o3d
from open3d._ml3d.torch.models.sparseconvnet import SparseConvUnet
from open3d._ml3d.torch.dataloaders.concat_batcher import SparseConvUnetBatch
from open3d._ml3d.utils import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from permanent_transient import SCANNET_20_BUCKET, Bucket  # noqa: E402


CKPT = ROOT / "checkpoints" / "sparseconvunet_scannet.pth"
CFG_NAME = "sparseconvunet_scannet.yml"
NUM_SEM_CLASSES = 20  # ScanNet-20 (model emits 20 logits; ignored label = 0)


def build_pt_lut() -> torch.Tensor:
    """LUT mapping a *valid* semantic class index (0..19, after the ignored-0
    shift) to {0: permanent, 1: transient}."""
    lut = torch.zeros(NUM_SEM_CLASSES, dtype=torch.long)
    for cid, bucket in SCANNET_20_BUCKET.items():
        if cid == 0:
            continue  # ignored label
        valid_idx = cid - 1  # shift to model output space
        lut[valid_idx] = 0 if bucket == Bucket.PERMANENT else 1
    return lut


class PTHead(nn.Module):
    """A tiny 2-class head that re-buckets the semantic logits.

    Two design options were possible:
      (a) tap penultimate features (richer signal, more params)
      (b) operate on the 20 semantic logits directly (cheap, interpretable)

    We pick (b) for the scaffold — it's a linear layer over 20 logits, which
    lets the head learn corrections to the hard-coded mapping (e.g., learn
    that a fridge in a fully-built-in alcove is closer to "counter" behavior).
    On Day 2 we can swap to (a) by patching SparseConvUnet to expose features
    pre-linear.
    """

    def __init__(self, num_sem_classes: int = NUM_SEM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(num_sem_classes, 2)

    def forward(self, sem_logits: torch.Tensor) -> torch.Tensor:
        return self.fc(sem_logits)


class SyntheticScenes(Dataset):
    """Random fake "rooms" — only here so --quick can verify wiring."""

    def __init__(self, n_scenes: int = 4, points_per_scene: int = 2000):
        self.n = n_scenes
        self.pps = points_per_scene
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx)
        pts = (rng.random((self.pps, 3), dtype=np.float32) - 0.5) * 4.0
        feat = rng.random((self.pps, 3), dtype=np.float32)
        # Random valid semantic labels in [1, 20]; 0 (ignored) excluded
        labels = rng.integers(1, NUM_SEM_CLASSES + 1, size=self.pps, dtype=np.int64)
        return {
            "point": torch.from_numpy(pts),
            "feat": torch.from_numpy(feat),
            "label": torch.from_numpy(labels.astype(np.int32)),
        }


def preprocess_one(model: SparseConvUnet, sample: dict, rng: np.random.Generator):
    """Replicates SparseConvUnet.preprocess for a single sample, but returns
    the keep-mask so we can re-index labels alongside points."""
    pts = sample["point"].numpy().astype(np.float32) / model.cfg.voxel_size
    feat = sample["feat"].numpy().astype(np.float32)
    labels = sample["label"].numpy().astype(np.int64)
    grid = model.cfg.grid_size
    m, M = pts.min(0), pts.max(0)
    offset = (
        -m
        + np.clip(grid - M + m - 0.001, 0, None) * rng.random(3)
        + np.clip(grid - M + m + 0.001, None, 0) * rng.random(3)
    )
    pts += offset
    keep = (pts.min(1) >= 0) & (pts.max(1) < grid)
    pts = pts[keep]
    feat = feat[keep]
    labels = labels[keep]
    pts = (pts.astype(np.int32) + 0.5).astype(np.float32)
    return {
        "data": {
            "point": torch.from_numpy(pts),
            "feat": torch.from_numpy(feat),
            "label": torch.from_numpy(labels.astype(np.int64)),
        }
    }


def collate(samples_with_data):
    batch = SparseConvUnetBatch(samples_with_data)
    return batch


def load_pretrained(device: str) -> SparseConvUnet:
    cfg_path = Path(o3d.__file__).parent / "_ml3d" / "configs" / CFG_NAME
    cfg = Config.load_from_file(str(cfg_path))
    model_cfg = {k: v for k, v in cfg.model.items() if k != "name"}
    model = SparseConvUnet(**model_cfg)
    ckpt = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    return model


def real_dataset(_root: str):
    raise NotImplementedError(
        "Day-2 TODO: wire in ScanNet train split. Pointcept's `ScanNet` dataset "
        "class is the cleanest source; export points/colors/labels to npy per "
        "scene and load here. Until then, use --quick."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="1 epoch, tiny synthetic data, CPU — smoke test only")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="SparseConvUnet eats memory fast; 1 is safe on CPU")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pt-weight", type=float, default=0.5,
                    help="lambda for the permanent/transient auxiliary loss")
    ap.add_argument("--scannet-root", default=None,
                    help="(Day 2) path to a prepared ScanNet train split")
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints" / "finetune"))
    ap.add_argument("--log-dir", default=str(ROOT / "runs"))
    args = ap.parse_args()

    device = "cpu"  # MPS lacks some sparse-conv kernels; CPU is the safe default
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir / time.strftime("%Y%m%d-%H%M%S")))

    print(f"Device: {device}")
    print("Loading pretrained SparseConvUnet (ScanNet-20)...")
    sem_model = load_pretrained(device)

    pt_head = PTHead().to(device)
    pt_lut = build_pt_lut().to(device)

    # Freeze the backbone in --quick (the scaffold's job is just to verify
    # the head trains end-to-end). On Day 2, unfreeze and use a lower LR
    # for the backbone vs. the new head.
    for p in sem_model.parameters():
        p.requires_grad = not args.quick

    params = list(pt_head.parameters()) + [
        p for p in sem_model.parameters() if p.requires_grad
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)

    if args.quick:
        epochs = 1
        dataset = SyntheticScenes(n_scenes=2, points_per_scene=1500)
    else:
        epochs = args.epochs
        if args.scannet_root is None:
            ap.error("--scannet-root is required without --quick")
        dataset = real_dataset(args.scannet_root)

    sem_loss_fn = nn.CrossEntropyLoss()
    pt_loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)

    print(f"Starting training: {epochs} epoch(s), {len(dataset)} scene(s) per epoch")
    global_step = 0
    for epoch in range(epochs):
        for i in range(len(dataset)):
            sample = dataset[i]
            batch = collate([preprocess_one(sem_model, sample, rng)])
            batch.to(device)
            labels = batch.label[0].to(device).long() - 1  # to 0..19
            # Drop any points whose label landed outside the valid range
            # (shouldn't happen with synthetic data, but cheap guard).
            valid = (labels >= 0) & (labels < NUM_SEM_CLASSES)
            if not valid.any():
                continue
            sem_logits = sem_model(batch)               # [N, 20]
            pt_logits = pt_head(sem_logits)             # [N, 2]
            pt_targets = pt_lut[labels.clamp_min(0)]

            sem_l = sem_loss_fn(sem_logits[valid], labels[valid])
            pt_l = pt_loss_fn(pt_logits[valid], pt_targets[valid])
            loss = sem_l + args.pt_weight * pt_l

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # Gradient-health signals — surfaced so a "loss prints but nothing
            # learns" bug shows up before we spend GPU time on it.
            head_grad_norm = pt_head.fc.weight.grad.norm().item() if pt_head.fc.weight.grad is not None else 0.0
            head_w_norm = pt_head.fc.weight.detach().norm().item()

            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/semantic", sem_l.item(), global_step)
            writer.add_scalar("loss/permanent_transient", pt_l.item(), global_step)
            writer.add_scalar("grad/pt_head_norm", head_grad_norm, global_step)
            print(f"  epoch {epoch} step {global_step}  "
                  f"loss={loss.item():.4f} (sem={sem_l.item():.4f} pt={pt_l.item():.4f})  "
                  f"head_grad={head_grad_norm:.3f} head_w={head_w_norm:.3f}")
            global_step += 1

        ckpt_path = out_dir / f"epoch_{epoch:03d}.pth"
        torch.save({
            "sem_state_dict": sem_model.state_dict(),
            "pt_head_state_dict": pt_head.state_dict(),
            "epoch": epoch,
        }, ckpt_path)
        print(f"  saved {ckpt_path}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
