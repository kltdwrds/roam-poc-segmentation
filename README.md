# Roam POC — 3D Semantic Segmentation for Permanent / Transient

A 48-hour artifact for a Roam application. Takes an indoor point cloud, runs
a semantic segmenter, and re-buckets the per-point predictions into
**permanent** (walls / floor / ceiling / doors / fixtures) vs **transient**
(furniture / clutter / appliances). Two days of work behind it: Day 1 is a
CPU pipeline runnable on an M1 (SparseConvUnet on ScanNet-20); Day 2 fine-tunes
**PTv3 on S3DIS** on a rented A100 with a joint permanent/transient auxiliary
head. **Headline result: binary permanent-vs-transient F1 = 96.98%** on the
held-out S3DIS Area-5 (68 scenes, single-pass val mode, no TTA).

Full writeup with design context, model decisions, and the three-pane viewer:
**Day 2 writeup → [Notion URL TK]**. The repo here is the supporting artifact
— code, checkpoints, eval results, and reproduce steps for a developer.

## File map

```
roam-poc-segmentation/
├── env/SETUP.md                install log + version pins (Day 1 + Day 2)
├── NOTES.md                    reviewer-facing technical notes (the interesting bits)
├── checkpoints/
│   ├── sparseconvunet_scannet.pth   Day-1 pretrained (Open3D-ML model zoo)
│   ├── ptv3_s3dis.pth               Day-2 pretrained (Pointcept model zoo, see NOTES.md)
│   └── finetune/ptv3_pt_epoch{00..02}.pth   Day-2 fine-tuned (best: epoch02, 177 MB ea)
├── eval/
│   ├── results.json            combined per-epoch metrics + winner
│   ├── results_epoch{00..02}.json   per-ckpt detail (per-class IoU + binary CM)
│   └── confusion.png           perm/transient confusion matrix for the winner
├── outputs/
│   ├── scene_{raw,semantic,permanent}.ply         Day-1 Redwood living-room
│   ├── day2_pretrained/Area_5-office_1/...        Day-2 broken-pretrained sanity
│   └── day2_final/{Area_5-conferenceRoom_1, Area_5-office_3}/...   Day-2 fine-tuned
├── src/
│   ├── permanent_transient.py  ScanNet-20 AND S3DIS-13 → {permanent, transient}
│   ├── infer.py                --model {sparseconv,ptv3}; export 3 PLYs per scene
│   ├── ptv3_infer.py           PTv3 backbone + Pointcept val transform pipeline
│   ├── finetune.py             Day-2 joint sem + perm/transient PTv3 fine-tune
│   ├── finetune_day1.py        Day-1 CPU smoke-test scaffold (kept for reproducibility)
│   ├── eval.py                 single-ckpt eval on Area-5
│   ├── eval_summary.py         combine 3-ckpt results + emit confusion.png
│   └── export_ply.py           colored PLY helpers + budget subsample
└── viewer/
    ├── index.html              three-pane Three.js viewer w/ shared OrbitControls
    └── scenes/ → ../outputs/day2_final/   (symlink for http.server's no-`..` rule)
```

## Reproduce — Day 1 (CPU, M1-friendly)

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install 'torch==2.2.2' torchvision 'numpy<2' open3d plyfile trimesh tqdm pyyaml tensorboard
mkdir -p checkpoints && curl -L -o checkpoints/sparseconvunet_scannet.pth \
  https://storage.googleapis.com/open3d-releases/model-zoo/sparseconvunet_scannet_202105031316utc.pth
python src/infer.py                         # SparseConvUnet on Redwood frag, 3 PLYs out
python src/finetune_day1.py --quick         # CPU smoke test, ~10s
```

## Reproduce — Day 2 (CUDA, A100-grade GPU)

Full step-by-step + version pins in `env/SETUP.md`. Short form:

```bash
# 1. env — uv venv 3.10 at /workspace/pointcept-env; install torch 2.5+cu124,
#    spconv-cu124, pointops (local build, sm_80), flash-attn 2.8, Pointcept HEAD
#    via PYTHONPATH=/workspace/Pointcept. See env/SETUP.md for the exact commands.

# 2. pretrained PTv3 S3DIS Area-5 rpe checkpoint (530 MB; see NOTES.md before trusting it!)
curl -L -o checkpoints/ptv3_s3dis.pth \
  https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth

# 3. preprocessed S3DIS (2 GB compressed, ~8 GB extracted; no auth)
mkdir -p /workspace/data && curl -L https://huggingface.co/datasets/Pointcept/s3dis-compressed/resolve/main/s3dis.tar.gz | tar -xzf - -C /workspace/data

# 4. fine-tune (3 epochs, ~16 min on A100; saves 3 lean ckpts at 177 MB each)
PYTHONPATH=/workspace/Pointcept python src/finetune.py --epochs 3 --batch-size 4 --amp

# 5. eval all 3 ckpts on Area-5; pick best by binary F1
for e in 00 01 02; do
  PYTHONPATH=/workspace/Pointcept python src/eval.py \
    --ckpt checkpoints/finetune/ptv3_pt_epoch${e}.pth --no-amp
done
PYTHONPATH=/workspace/Pointcept python src/eval_summary.py    # writes eval/results.json + confusion.png

# 6. export final PLYs from clean held-out scenes (uses best ckpt)
for s in conferenceRoom_1 office_3; do
  PYTHONPATH=/workspace/Pointcept python src/infer.py --model ptv3 --finetuned \
    --ckpt checkpoints/finetune/ptv3_pt_epoch02.pth --scene-id Area_5/$s
done

# 7. viewer
python -m http.server -d viewer/ 8080      # http://localhost:8080 in Chrome
```

## Class-mapping rationale (short)

Two label spaces are supported, with separate but parallel permanent/transient
bucket maps in `src/permanent_transient.py`. `permanent_mask()` auto-detects
the space by id range — ScanNet-20 (ids 0..20) vs S3DIS-13 (ids 0..12). The
guiding principle for both: PERMANENT = anything defining the building shell
(architecture + plumbed fixtures); TRANSIENT = anything sittable / liftable
/ plug-in-able / seasonal. Full per-class reasoning in the source comments.

## Honest caveats

- **3 epochs vs the official 3000-epoch S3DIS training schedule.** Our 64% mIoU
  on Area-5 is below the official PTv3 paper's 73.6% TTA mIoU number; the
  binary perm/transient F1 (97%) is the demo-relevant headline, and that's
  insensitive to the per-class long tail.
- **The released pretrained checkpoint was unexpectedly broken** (9.6% mIoU on
  Area-5 vs the 73.6% it claims) — we fine-tuned on top of it anyway because
  one epoch was enough to realign the seg_head. Full investigation in NOTES.md.
- **Eval is single-pass, not TTA.** Pointcept's official tester does 10-augmentation
  TTA aggregation; we don't. Direct comparison to the paper's headline mIoU
  isn't apples-to-apples.
- The held-out scene for the viewer is a single S3DIS Area-5 room (`conferenceRoom_1`
  or `office_3`). For the demo, that's enough; for production, you'd want
  cross-Area evaluation + open-vocab inputs.

## References

- **Pointcept** (Day 2 codebase, S3DIS prep, PTv3): <https://github.com/Pointcept/Pointcept>
- **Point Transformer V3** (Wu et al., CVPR 2024): <https://arxiv.org/abs/2312.10035>
- **Open3D-ML** (Day 1 model + I/O): <https://github.com/isl-org/Open3D-ML>
- **S3DIS** (preprocessed mirror): <https://huggingface.co/datasets/Pointcept/s3dis-compressed>
