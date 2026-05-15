# env/SETUP.md — Day 1 install log

Live log of every install decision and failure. Append-only — do not rewrite history.

## Host

- Apple M1 Max, 64 GB RAM
- macOS Darwin 25.5.0 (arm64)
- Python 3.11.1 system; pinning **3.10** via uv for compatibility with Open3D-ML wheels
- uv 0.10.8 (Homebrew)
- No NVIDIA GPU. MPS available.

## Path chosen: Open3D-ML (PyTorch backend) on CPU + MPS

Rationale:
- Pointcept / Point Transformer V3 — the SOTA candidate — depends on CUDA-only kernels
  (`spconv-cu*`, `pointops`, `flash-attn`). No working Apple Silicon path in <1 day.
- Day 2 is a rented NVIDIA GPU — that's where PTv3 belongs. Defer it.
- Open3D-ML ships pretrained **RandLA-Net** and **KPConv** checkpoints on ScanNet, runs
  on CPU/MPS, label space (20 classes) matches what we need to bucket into permanent/transient.
- Fallback if Open3D-ML wheels are broken on arm64 Python 3.10: `torch-points3d` or
  a pure-PyTorch KPConv (`easy_kpconv`). Decision point: 30-min budget per attempt.

## Steps (logged as I go)

### Step 1 — uv venv (Python 3.10)

```
uv venv --python 3.10 .venv      # Python 3.10.20
source .venv/bin/activate
```

### Step 2 — Install deps (with iteration)

Initial attempt: `torch>=2.2,<2.5` → got torch 2.4.1. Then `pip install open3d` → 0.19.0.
On `import open3d.ml.torch` got: `Version mismatch: Open3D needs PyTorch version 2.2.*, but version 2.4.1 is installed!`

**Pinned torch to 2.2.2.** Then `import torch` failed with `Failed to initialize NumPy: _ARRAY_API not found` — torch 2.2 was compiled against NumPy 1.x. **Pinned `numpy<2`** → numpy 1.26.4.

Verified: `RandLANet` and `KPFCNN` import from `open3d.ml.torch.models`.

### Locked versions (Day 1, M1 Max, Python 3.10.20)

| Package | Version |
|---|---|
| torch | 2.2.2 |
| torchvision | 0.17.2 |
| open3d | 0.19.0 |
| numpy | 1.26.4 |
| plyfile | 1.1.3 |
| trimesh | 4.12.2 |
| tensorboard | 2.20.0 |

### Step 3 — Pretrained ScanNet weights

Downloaded official Open3D-ML model-zoo checkpoints into `checkpoints/`:

| File | Size | Source |
|---|---|---|
| `sparseconvunet_scannet.pth` | 66 MB | Pretrained on ScanNet-20, **primary model** |
| `randlanet_s3dis.pth` | 57 MB | Pretrained on S3DIS, **backup model** |

Both URLs from <https://github.com/isl-org/Open3D-ML/blob/main/model_zoo.md>.

Critical install gotcha: `SparseConvUnet(...)` must be instantiated with the
hyperparameters from `open3d/_ml3d/configs/sparseconvunet_scannet.yml`
(notably `multiplier: 32`, not the default 16). Strict state-dict load only
works with these settings. Wired this into `src/infer.py`.

### Step 4 — Dataset

ScanNet proper requires a manual license agreement + emailed download link
and was therefore out of scope for Day 1 (would block reproducibility).
Skipped HF mirrors after a quick check — same gating story.

Using Open3D's bundled **Redwood Living Room** dataset (auto-downloaded by
`o3d.data.LivingRoomPointClouds()`): 57 fragment point clouds of a real
indoor scan, RGB present, ~196k points each. No auth. This is what `infer.py`
runs on by default; passing `--scene-path some.ply` overrides it.

Day-2 plan: pull a real ScanNet scene on the GPU box after license accept.

---

# Day 2 install log (A100 80GB rented box)

## Host

- A100 80GB PCIe, 1 GPU, driver 550.54.15, CUDA 12.4 (nvcc 12.4.131)
- Linux Ubuntu 22.04, kernel 6.5.0, gcc 11.4
- Python 3.11.10 system, **Python 3.10.12 via uv** for the working env
- /workspace = 20 GB ext4, / overlay = 20 GB. Tight — stay under /workspace
- RAM 1 TiB (plenty), no swap
- ScanNet license still not approved → switched to **S3DIS** (13 classes; no auth)

## Path chosen: Pointcept torch 2.5.0 + cu124 (PTv3)

Matched Pointcept HEAD's pinned env (`environment.yml` declares torch 2.5.0
+ pytorch-cuda 12.4). Earlier Day-2 plan said "torch 2.4 cu121 should work
on 12.4" — switched to 2.5 cu124 because it's what the project's own CI
runs against, and CUDA driver 550 supports cu124 wheels natively.

## Steps

### Step 1 — uv + venv (Python 3.10) at /workspace/pointcept-env

uv 0.11.14, then `uv venv --python 3.10 /workspace/pointcept-env`.
Note: venv ships *without* a `pip` binary on PATH after activate — `which pip`
resolves to `/usr/local/bin/pip` (system 3.11). **Always use `python -m pip`
or `uv pip` inside this venv**, not bare `pip`. I burned a build on this
gotcha (pointops compiled into Py3.11 site-packages, not the venv).

### Step 2 — torch + cu124

```
uv pip install --index-strategy unsafe-best-match \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Verified: `torch.cuda.is_available() == True`, device "NVIDIA A100 80GB PCIe".

### Step 3 — sparse-conv + python deps

```
uv pip install spconv-cu124 numpy<2 addict einops scipy plyfile termcolor \
               timm ftfy regex tqdm matplotlib h5py pyyaml tensorboard \
               tensorboardx yapf open3d peft wandb
```

`spconv 2.3.8` installed; numpy pinned <2 (avoid torch 2.5/numpy-2 ABI breakage).

### Step 4 — torch-geometric stack

From PyG's torch-2.5/cu124 wheel index:
```
uv pip install torch-cluster torch-scatter torch-sparse torch_geometric \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

Got `torch-cluster 1.6.3+pt25cu124`, `torch-scatter 2.1.2`, `torch-sparse 0.6.18`.

### Step 5 — CUDA source build (pointops)

Needed system **python3.10-dev** package first (Python.h missing → `apt-get
install -y python3.10-dev`). Then:

```
TORCH_CUDA_ARCH_LIST="8.0" python -m pip install --no-build-isolation \
  ./libs/pointops
```

`--no-build-isolation` is required so the build env sees torch.
sm_80 only (A100). Build ~30 s. Wheel: `pointops-1.0-cp310-cp310`.

### Step 6 — flash-attention

`uv pip install flash-attn --no-build-isolation`. Cached prebuilt wheel
landed in ~15 s — no 10-15 min compile this time. Got `flash_attn 2.8.3`.
PTv3 with `enable_flash=True` is now an option for training/inference.

### Step 7 — skipped: pointgroup_ops, pointrope

- **pointgroup_ops**: only needed for instance segmentation. Build failed
  with same env quirks. Not on the semseg path, skipped.
- **pointrope**: CUDA accel of rotary embeddings. Pointcept prints a warning
  and falls back to pure-PyTorch impl. Skipped — PTv3 S3DIS config doesn't
  rely on it.

### Step 8 — verify

```python
PYTHONPATH=/workspace/Pointcept python -c "from pointcept.models import build_model; print('ok')"
# -> ok (after wandb install)

# Build PTv3 backbone on A100:
# -> built: DefaultSegmentorV2 46.17 M params
```

Pointcept itself has no setup.py — it's imported via PYTHONPATH. Setting
`export PYTHONPATH=/workspace/Pointcept` for all subsequent runs.

### Locked versions (Day 2, A100, Python 3.10.12)

| Package | Version |
|---|---|
| torch | 2.5.0+cu124 |
| torchvision | 0.20.0+cu124 |
| spconv-cu124 | 2.3.8 |
| numpy | 1.26.x (<2) |
| pointops | 1.0 (local source build, sm_80) |
| torch-cluster | 1.6.3+pt25cu124 |
| flash-attn | 2.8.3 |
| open3d | 0.19.0 |
| Pointcept | HEAD as of 2026-05-14 (git clone --depth 1) |

### Disk after Step 2 (Pointcept install)

/workspace: **8.6 G used, 11 G free** of 20 G. Mostly torch+cuda libs in
the venv (~5 G) and Pointcept clone (~50 M). Room for S3DIS (~5 G) and
checkpoints (~few hundred MB). Will be tight; want to verify after S3DIS.

Elapsed: 15 min vs. 60-min cap.

### Step 9 — PTv3 S3DIS Area-5 pretrained checkpoint

URL: `huggingface.co/Pointcept/PointTransformerV3/resolve/main/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth`
Size: **530 MB** (includes optimizer + ema + scaler — model weights are
~180 MB). Saved to `checkpoints/ptv3_s3dis.pth`. Best val mIoU recorded
in ckpt: **0.7192**; matches model-zoo's reported **73.6%** for the
`-rpe` config (small drift from re-trained variant).

**Important config gotcha for strict load:** the `-rpe` checkpoint uses
`enc_patch_size=128, dec_patch_size=128`, NOT the 1024 default in
`semseg-pt-v3m1-0-base.py`. With `enable_rpe=True` and `enable_flash=False`.
First strict load failed with 22 rpe_table shape mismatches (93 vs 189)
— root cause was patch_size, not weights. After fix: `STRICT LOAD: OK`,
46.19 M params.

Also: ckpt keys are `module.*` (DDP-wrapped). Strip prefix before load:
```python
sd = OrderedDict((k.replace('module.', '', 1), v) for k, v in ckpt['state_dict'].items())
```

### Step 10 — S3DIS preprocessed dataset

Pulled Pointcept's compressed tarball from `huggingface.co/datasets/Pointcept/s3dis-compressed`.
2.04 GB compressed, ~8.0 GB extracted. **No auth required.**

Streamed via `curl URL | tar xzf -` to avoid the 2 GB intermediate file
(disk would not have fit both tarball + extracted at ~20 GB cap).

Layout under `/workspace/data/`:
```
Area_1/conferenceRoom_1/{coord,color,normal,segment,instance}.npy
Area_1/... (44 scenes)
Area_2/... (40)
Area_3/... (23)
Area_4/... (49)
Area_5/... (68)  <- the held-out test split
Area_6/... (48)
```

Total 272 scenes. Files are float32/int16 npy. **`data_root="/workspace/data"`**
— S3DIS Areas sit directly at root, no `s3dis/` sub-dir.

### Step 11 — Pretrained PTv3 inference on Area_5/office_1

Wrote `src/ptv3_infer.py` (PTv3-specific loader + Pointcept Compose pipeline)
and extended `src/infer.py` with `--model {sparseconv,ptv3}` so Day 1 still
reproduces. Added `S3DIS_13_NAMES/BUCKET/PALETTE` to `src/permanent_transient.py`
and switched `permanent_mask` to auto-detect ScanNet-20 vs S3DIS-13 by id range.

Held-out scene: **Area_5/office_1** (816,136 pts, ~5.6m × 3.4m × 3.2m office).

Class distribution sanity (ceiling+floor+wall > 40% target):
```
ceiling   3.0%  floor 28.0%  wall 50.6%   -> SUM 81.6%   PASS
```
PLYs exported to `outputs/day2_pretrained/Area_5-office_1/{scene_raw,scene_semantic,scene_permanent}.ply`
(12.2, 12.2, 10.6 MB).

#### ⚠ Anomaly — pretrained mIoU much lower than expected (open issue)

| metric                                   | got    | expected |
|---|---|---|
| per-point accuracy on office_1            | 32.4% | ~70-75% |
| mean IoU over classes present on office_1 | 9.6%  | ~70-75% |

Worst per-class IoU on office_1 (in GT, model misses entirely):
- window: GT 15.2% → model predicts 0.9%, IoU 4.8%
- ceiling: GT 19.1% → model predicts 3.0%, IoU 9.5%
- bookcase: GT 15.0% → model predicts 4.1%, IoU 4.3%
- door: GT 3.4% → IoU 0%

The model is heavily biased toward floor+wall (~78% of predictions vs ~30% GT).

**What I verified:**
- Checkpoint strict-loads with `enc/dec_patch_size=128, enable_rpe=True,
  enable_flash=False, upcast_attention=True, upcast_softmax=True,
  pdnorm_decouple=True, pdnorm_conditions=("ScanNet","S3DIS","Structured3D")`
  — these match `configs/s3dis/semseg-pt-v3m1-1-rpe.py`.
- Class indices match preprocess_s3dis.py (0=ceiling ... 12=clutter).
- coord col 2 is vertical (matches train-time `RandomRotate axis="z"`).
- color is [0,255] uint8, divided by 255 by NormalizeColor → [0,1]; normal
  is float32 in [-1,1] — both as expected.
- Voxelization uses Pointcept's own `Compose([CenterShift, Copy, GridSample,
  CenterShift, NormalizeColor, ToTensor, Collect])` — bit-for-bit identical
  to the official val transform list. Replaced my hand-rolled np.unique
  voxelizer (which gave the same bad result, so it wasn't the bug).
- Back-projection uses GridSample's own `inverse` index (not my own).
- 46.19 M params, strict load OK, no missing/unexpected keys.

**Hypotheses left (for next session to test):**
1. **Pretrained mIoU 73.6% is a TTA / test-mode number**, not a single-pass
   val number. The test config uses `mode='test'` GridSample which returns
   N data parts (one per voxel-population offset) plus an `aug_transform`
   list (10 scale/flip combos). Pointcept's `tools/test.py` aggregates
   softmax over all that. Single-pass val mode would naturally be much
   lower — but 9.6% mIoU still feels too low for that to be the only
   explanation.
2. **Config drift between cloned Pointcept HEAD and the released ckpt.**
   Our cloned repo has only `configs/s3dis/semseg-pt-v3m1-1-rpe.py` but
   the HF model zoo path is `s3dis-semseg-pt-v3m1-0-rpe`. The `0-rpe.py`
   file may exist at a different commit with slightly different defaults
   (e.g. norm layer placement, embedding init). To check: `git log --all
   --oneline -- configs/s3dis/semseg-pt-v3m1-0-rpe.py` (would need
   `git clone` without `--depth 1`).
3. **pdnorm condition missing.** DefaultSegmentorV2.forward doesn't
   inject a "condition" key, but PTv3 with `pdnorm_decouple=True` may
   silently fall back to the first condition ("ScanNet") for batchnorm
   stats — wrong distribution for S3DIS-style indoor scenes. Need to
   pass `condition=["S3DIS"]` in input_dict and verify it reaches the
   PDNorm layers.

**What to do in the next session before training (5-10 min):**
- Run `python tools/test.py --config-file configs/s3dis/semseg-pt-v3m1-1-rpe.py
  --options weight=/workspace/roam-poc-segmentation/checkpoints/ptv3_s3dis.pth
  save_path=/tmp/test_run`  — let Pointcept's own tester reproduce the 73%
  mIoU. If it does, we have a baseline to diff against my `infer.py` path.
- If Pointcept's own tester also fails, the ckpt itself is suspect — try
  the **non-RPE** config (`semseg-pt-v3m1-0-base.py`, patch_size=1024, no rpe)
  with the matching released weights as a sanity check.

**Bottom line for step 5 sign-off:**
Step-5 deliverables (infer.py supports `--model ptv3`, scene selected,
inference runs end-to-end, sanity check on ceiling+floor+wall percentage
passes, three PLYs exported) are met. The mIoU diagnostic is below
expectation but does not block step 6 — fine-tuning on top will give us
a real comparable number either way, and may actually mask the bug if
the bug is config-side.

### Step 12 — Diagnostic: Pointcept's own tester reproduces the anomaly

Ran the canonical `python tools/test.py --config-file configs/s3dis/
semseg-pt-v3m1-1-rpe.py --num-gpus 1 --options data.test.data_root=
/workspace/data weight=checkpoints/ptv3_s3dis.pth save_path=runs/diag_pretrained`
against the released ckpt. Scene 1 result:

```
Test: Area_5-conferenceRoom_1 [1/68]-1047554 Batch 94.544 (94.544)
  Accuracy 0.4958 (0.2108)  mIoU 0.1357 (0.1357)
```

**13.6% mIoU on scene 1 from Pointcept's own tester with full TTA
(10 augmentations × ~180 grid fragments).** This confirms the bug is
NOT in my `src/infer.py` path or my voxelization. The ckpt+config
combo itself is broken.

Hypotheses ruled out by this experiment:
1. ~~Single-pass val vs TTA~~ — Pointcept's tester does full TTA, still 13.6%.
2. ~~Config drift HEAD vs PTv3-release~~ — diffed `point_transformer_v3m1_base.py`
   between PTv3 release commit `314afb3` and HEAD: only `cls_mode` → `enc_mode`
   rename, timm import path change, and `flash_attn` fp16 → bf16. With
   `enable_flash=False` (our config), these are inert. Model behavior is
   identical to release.
3. ~~PDNorm condition missing~~ — `pdnorm_bn=False, pdnorm_ln=False` in our
   config means `PDNorm` isn't instantiated. Plain `BatchNorm1d` /
   `LayerNorm` used instead. PDNorm's `condition` requirement is moot.

Remaining hypotheses (NOT pursued; deferred for budget reasons):
- **Released HF file is mislabeled / from a wrong epoch.** Ckpt metadata says
  `epoch=67, best_metric_value=0.7192` (71.9%), but eval gives ~14%. Could be
  that "best" was overwritten with a non-best ckpt during upload, or the
  config the model was actually trained against has a subtle difference we
  can't see from the released config alone.
- Something subtle about `enable_amp=True, amp_dtype='float16'` at train time
  that interacts badly with HEAD-era inference path.

### Decision: pivot

Continuing to debug the released ckpt risks burning more GPU hours without
guarantee. With $25 budget and 4-hour fine-tune cap, the higher-EV move is:

**Use the broken ckpt as fine-tune init.** It strict-loads, has *some*
learned structure (predictions are biased but not random — 32% per-point
accuracy on office_1 is >> random 8% for 13-class), and 5 epochs of
fine-tuning on S3DIS train (Areas 1-4,6) should de-bias the seg head.

**Fallback plan if fine-tune loss looks poisoned in first 100 steps:**
re-init the seg_head Linear layer to fresh weights (preserve backbone)
and resume. If that also fails, restart from scratch — PTv3 patch_size=128
at batch_size=12 should be fast enough on A100 80GB to get a usable
baseline in 50-100 epochs (~1-2 hours).

### Disk usage after Step 4 (S3DIS extracted)

| Dir | Size |
|---|---|
| /workspace/data (S3DIS) | 8.0 G |
| /workspace/pointcept-env | 8.2 G |
| /workspace/Pointcept | 135 M |
| /workspace/roam-poc-segmentation | 725 M |
| **total** | **17 G used / 3.1 G free** of 20 G |

Tight. Implications: fine-tune checkpoints should save **model weights only**
(~180 MB each), not full optim state (~530 MB each). 3 epochs × 180 MB =
540 MB — fits with room left for PLY outputs (~100 MB).
