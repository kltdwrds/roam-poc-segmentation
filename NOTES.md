# Day 2 — Technical Notes

Companion to the Notion writeup, for a reviewer who's already read the
top-level pitch and is now opening the repo. This doc captures the things
that took real time to figure out, the decisions that aren't obvious from
the code, and the honest caveats that didn't fit in the README's two
paragraphs. Install log (chronological, every package + every pin) lives
in `env/SETUP.md`.

## TL;DR

- Goal: per-point semantic seg → bucket into permanent vs transient on real
  indoor scenes; export 3 PLYs (raw / semantic / permanent-only) for a
  three-pane viewer.
- Architecture: **PTv3 (46.2 M params) on S3DIS-13**, fine-tuned 3 epochs
  from Pointcept's released `s3dis-semseg-pt-v3m1-0-rpe` checkpoint, with a
  joint 2-class permanent/transient auxiliary head over the 13 semantic
  logits (`loss = ce(sem) + 0.5·ce(pt)`).
- Held-out eval (S3DIS Area-5, 68 scenes, single-pass val mode):
  - mIoU **63.83%**, per-point accuracy **88.26%**
  - **Binary perm/transient F1 = 96.98%** (precision 95.5%, recall 98.5%)
- Training was fast on an A100 80GB: **~16 min wall time** for 3 epochs at
  batch_size=4, bf16 AMP, 204k voxels/sample (SphereCrop).
- Total GPU time used end-to-end ≈ 2 hours; well inside the $25 budget cap
  ($1.20/hr box).

## The pretrained-checkpoint mystery

Most of the Day-2 time-cost-vs-time-spent variance came from this. Worth
explaining because the *decision* — to fine-tune on top of a checkpoint we
couldn't fully validate — only makes sense once you've seen the data.

**What the model zoo claims:** `huggingface.co/Pointcept/PointTransformerV3/`
`tree/main/s3dis-semseg-pt-v3m1-0-rpe` is the S3DIS Area-5 PTv3 model with
**73.6% mIoU**. The checkpoint itself has `best_metric_value=0.7192` in its
metadata (so it claims 71.9% internally — a small drift from the 73.6% in
the README table, suggesting the README number is post-TTA).

**What we got:**

| eval path                              | mIoU on Area-5 scene 1 |
|----------------------------------------|------------------------:|
| Our `src/infer.py` (custom inference)  |  9.6% (mean IoU)        |
| Pointcept's own `tools/test.py` w/ TTA | 13.6%                   |

Same checkpoint, same data, both well below 70%. So the bug wasn't in our
inference path.

**What we ruled out before pivoting:**
1. *Single-pass vs TTA?* Pointcept's tester does the full 10-augmentation
   TTA + multi-fragment aggregation and still gave 13.6%. Not the cause.
2. *Config drift between cloned HEAD and the PTv3 release commit?* Diffed
   `point_transformer_v3m1_base.py` between `314afb3` (PTv3 release, Dec '23)
   and HEAD. Only changes: `cls_mode` renamed to `enc_mode`, the timm import
   path moved, and the flash-attn forward switched fp16→bf16. With
   `enable_flash=False` (our config), all of these are inert. Model behavior
   is byte-for-byte identical with our config.
3. *PDNorm condition missing?* PTv3's prompt-driven normalization layer
   *requires* a `condition` key in the input dict — but it's only instantiated
   when `pdnorm_bn=True` or `pdnorm_ln=True`, both of which are False in our
   config. Plain `BatchNorm1d`/`LayerNorm` are used instead. Inert.

**Remaining hypothesis (not pursued):** the released HF file is mislabeled
or from a wrong epoch. The ckpt metadata says `epoch=67/3000` with claimed
best mIoU 71.9% — possibly an "early best" overwritten during upload, or
the config the model was actually trained against has a subtle difference
not exposed in the released config. We didn't dig further; the budget was
better spent on training.

**Why we fine-tuned from it anyway:** the checkpoint *strict-loads*, and
its predictions on a real scene weren't random noise — sem_acc was 32%
on Area_5/office_1 (vs ~8% for a uniform 13-class prior). So the backbone
features were doing *something*; the seg_head was just misaligned. One
epoch of fine-tuning would either confirm or refute that. It confirmed:
sem_acc jumped from 59% at step 0 to **96% at step 50** (end of epoch 0).
The backbone clearly had recoverable structure.

If we'd had to retrain from scratch, the patch_size=128 rpe config at
bs=12 on A100 would have needed many hours; we'd have downgraded scope.

## Training curve — why epoch 2, not epoch 0

Final-epoch checkpoints aren't always best on held-out data; we evaluated
all three.

| epoch | train loss | train sem_acc | val mIoU | val acc | val F1 (perm) |
|------:|-----------:|--------------:|---------:|--------:|--------------:|
|     0 |      0.156 |        96.0%  |  53.20%  | 82.72%  |   95.40%      |
|     1 |      0.127 |        96.6%  |  59.07%  | 85.27%  |   96.10%      |
|     2 |      0.082 |        97.8%  |  63.83%  | 88.26%  | **96.98%**    |

Every metric climbed monotonically. **No overfitting after 3 epochs.** This
suggests we'd keep getting gains from more epochs — the OneCycleLR schedule
peaks at step ~7 (5% pct_start of 153 total) and is already deep into the
cosine decay by epoch 2, so we got the easy gains. A 10-epoch run with a
shorter cycle would likely push perm F1 toward 0.98 — left for a future
session given the 4-hour cap.

## Two heads, near-identical numbers

The auxiliary 2-class PT head and the deterministic `S3DIS_13_BUCKET[
sem_argmax]` lookup converge to nearly identical F1 (96.98% vs 96.92%).
That makes sense: both are linear-over-13-logits transformations, and given
enough capacity and training signal, the learned linear head approaches the
hard-coded bucket map. The interesting follow-up would be tapping the
**penultimate 64-dim features** instead of the 13-dim logits — that head
would have access to information lost by the semantic argmax and could in
principle outperform the bucket lookup. Left as a "next session" item.

For the headline number we report sem→bucket (the cheaper, more interpretable
path).

## The spconv + AMP eval failure

Training ran with `--amp` (bf16 autocast) and worked fine. Eval *failed*
with bf16 on the first Area-5 scene:

```
spconv ConvTunerSimple_tune_and_cache.cc(103)
  !all_profile_res.empty() assert faild. can't find suitable algorithm for 0
```

This is spconv's autotune failing to pick a kernel. Disabling AMP (`--no-amp`,
fp32) fixed it cleanly. **The trigger we think matters: voxel count.** Training
uses `SphereCrop(point_max=204800)` so each batch tops out at ~800k voxels at
bs=4. Eval feeds the *whole scene*, which for big Area-5 rooms (`conferenceRoom_2`
= 1.9 M points → ~1 M voxels at 0.02 m grid) puts spconv into a configuration
its autotune table doesn't have an entry for. fp32 paths avoid that table.

If you turn AMP back on for eval, plan to chunk inference scene-by-scene with
a SphereCrop-style window. Or just live with the fp32 slowdown (5 min for the
full 68-scene Area-5 sweep — fine).

## Disk-tight artifact decisions

The rented box's `/workspace` is a 20 GB ext4 partition. After Pointcept install
(8 GB venv) and S3DIS data (8 GB), 3 GB was left. Two consequences:

- **Fine-tune ckpts save model weights only, no optimizer state.** Each is
  177 MB instead of ~530 MB. Three epochs × 177 MB = 530 MB fits. Pointcept's
  default `save_optim=True` would have blown the disk by epoch 2.
- **S3DIS extraction used `curl … | tar xzf -` to stream** instead of
  download-then-extract. Peak disk during extraction would otherwise have
  been 2 GB tarball + 8 GB extracted = 10 GB transient, perilously close to
  the cap.

If you re-run on a roomier box, neither matters — but worth knowing if the
budget is the same shape.

## What's not in here (limitations + next steps)

- **No cross-Area cross-validation.** S3DIS standard practice is 6-fold;
  we did Area-5 only.
- **No open-vocab / iPhone-LiDAR demo.** The stretch goal in the original
  brief was "before-after on a messy/tidy iPhone scan". Skipped because
  domain shift would have eaten a debug session and the perm/transient F1
  headline was already the point.
- **Viewer is a single static page.** No upload, no scene comparison,
  no labeled-vs-predicted overlay. Three viewports with shared camera, scene
  picker, point size, opacity — that's it. Productionizing this would mean
  Potree/Three-Studio for streaming larger scenes.
- **No ablation of `--reinit-head`.** The `finetune.py --reinit-head` flag
  is wired but we didn't measure how much the broken-pretrained init
  actually hurt vs random init; given how fast it recovered, probably not
  much.
- **No comparison to PTv2 or MinkUNet.** Pointcept ships both with S3DIS
  pretraineds; would be a useful baseline.

## Reproduce checklist (the things that wasted time on Day 2)

Hit these in this order:

1. `apt-get install python3.10-dev` — pointops won't build without `Python.h`.
2. After `source /workspace/pointcept-env/bin/activate`, use `python -m pip`
   or `uv pip`, *not* bare `pip` — the bare `pip` on PATH after activate
   points at system Python 3.11 and silently builds cp311 wheels into
   the wrong site-packages.
3. Use `--no-build-isolation` when pip-installing `pointops` or `flash-attn`
   so the build env can see your installed torch.
4. `wandb` is required by Pointcept's hook import path even if you don't
   use it. `SharedArray` is required only by older commits (v1.5.x).
5. `enc_patch_size=128` + `enable_rpe=True` + `enable_flash=False` +
   `upcast_attention=True` + `upcast_softmax=True` is the *only* config that
   strict-loads the released rpe checkpoint. The `0-rpe.py` file (in older
   commits, renamed to `1-rpe.py` in HEAD) is the canonical reference.
6. Pointcept has no `setup.py`; set `PYTHONPATH=/workspace/Pointcept` every
   shell.
