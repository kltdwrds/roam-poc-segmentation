"""
ScanNet semantic class → {PERMANENT, TRANSIENT} mapping.

Bucketing rationale for the Roam Reconstruct demo: a Roam "permanent" point is
something that is unlikely to move between visits to the same space on a
days-to-weeks timescale, and is therefore safe to fuse across captures into a
stable 3D index. A "transient" point belongs to a movable / removable object
and should be evicted (or routed to a separate transient layer) rather than
baked into the long-lived map.

Cutoffs reflect the intent of the system, not the ScanNet annotators:
- Architecture & built-ins → PERMANENT
- Anything sittable, liftable, plug-in-able, or seasonal → TRANSIENT
- "otherfurniture" is the ScanNet catch-all (rugs, plants, decor, boxes …)
  and is treated as TRANSIENT — it's a noisy bucket but it's overwhelmingly
  removable stuff.
"""

from enum import Enum
from typing import Dict


class Bucket(str, Enum):
    PERMANENT = "permanent"
    TRANSIENT = "transient"


# The standard ScanNet-20 label set used by RandLANet / KPFCNN pretrained on
# ScanNet via Open3D-ML. Index 0 is reserved for "unlabeled".
SCANNET_20_NAMES: Dict[int, str] = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "desk",
    14: "curtain",
    15: "refrigerator",
    16: "shower curtain",
    17: "toilet",
    18: "sink",
    19: "bathtub",
    20: "otherfurniture",
}


# Authoritative bucket map.
# Keep this dict the single source of truth — infer.py, finetune.py, and
# export_ply.py all read from here.
SCANNET_20_BUCKET: Dict[int, Bucket] = {
    0: Bucket.TRANSIENT,        # unlabeled → treat as transient (safer default:
                                #   we don't want to fuse unknowns into the
                                #   permanent index)
    1: Bucket.PERMANENT,        # wall      — building shell
    2: Bucket.PERMANENT,        # floor     — building shell
    3: Bucket.PERMANENT,        # cabinet   — built-in cabinetry; debatable for
                                #   free-standing units, but most ScanNet
                                #   cabinets are wall-mounted kitchen/bath
    4: Bucket.TRANSIENT,        # bed       — movable
    5: Bucket.TRANSIENT,        # chair
    6: Bucket.TRANSIENT,        # sofa
    7: Bucket.TRANSIENT,        # table
    8: Bucket.PERMANENT,        # door      — part of building envelope
    9: Bucket.PERMANENT,        # window    — part of building envelope
    10: Bucket.TRANSIENT,       # bookshelf — usually free-standing in ScanNet
    11: Bucket.TRANSIENT,       # picture   — hung art, removable
    12: Bucket.PERMANENT,       # counter   — built-in countertop
    13: Bucket.TRANSIENT,       # desk
    14: Bucket.TRANSIENT,       # curtain   — fabric, swapped seasonally
    15: Bucket.TRANSIENT,       # refrigerator — plug-in appliance, replaceable
    16: Bucket.TRANSIENT,       # shower curtain
    17: Bucket.PERMANENT,       # toilet     — plumbed fixture; very stable
    18: Bucket.PERMANENT,       # sink       — plumbed fixture
    19: Bucket.PERMANENT,       # bathtub    — plumbed fixture
    20: Bucket.TRANSIENT,       # otherfurniture — catch-all, mostly movable
}

# Talking points for Arvin / interview:
#   1. Doors and windows go PERMANENT because Roam's spatial OS needs to
#      reason about openings (egress, light, navigation) — those don't change.
#   2. Plumbed fixtures (toilet/sink/bathtub) are PERMANENT despite being
#      "objects" — replacing one is a renovation event, not daily churn.
#      This shows the bucketing is *semantic* (likelihood of change), not
#      *geometric* (free-standing vs. attached).
#   3. Refrigerators are TRANSIENT — they're appliances that get replaced or
#      moved. Counters are PERMANENT because they're stone/laminate installs.
#      This is exactly the kind of distinction that makes a learned 2-class
#      head useful vs. a hard-coded list: the model can also learn that a
#      built-in fridge alcove behaves more like a counter.
#   4. "Unlabeled" defaulting to TRANSIENT is a conservative choice — we'd
#      rather under-fuse the permanent index than corrupt it.


def bucket_for_class(class_id: int) -> Bucket:
    """Map a ScanNet-20 class id to a permanent/transient bucket."""
    return SCANNET_20_BUCKET[class_id]


def permanent_mask(class_ids):
    """Return a boolean numpy array, True where the class is PERMANENT.

    Auto-detects whether ids are ScanNet-20 (0..20) or S3DIS-13 (0..12) by
    checking the max value. Both label spaces are supported because Day 1
    used ScanNet (CPU/Open3D-ML) and Day 2 uses S3DIS (GPU/Pointcept).
    """
    import numpy as np
    arr = np.asarray(class_ids)
    bucket_map = S3DIS_13_BUCKET if int(arr.max(initial=0)) < 13 else SCANNET_20_BUCKET
    return np.array(
        [bucket_map[int(c)] == Bucket.PERMANENT for c in arr],
        dtype=bool,
    )


# ============================================================================
# S3DIS-13 label set (used on Day 2 with Pointcept PTv3)
# ============================================================================
#
# Standard 13-class indoor scene parsing labels from the Stanford Large-Scale
# 3D Indoor Spaces dataset. Indices are 0-12 in Pointcept's preprocessed npy
# (no "unlabeled" wildcard — every point is annotated).
#
# Rationale carries over from ScanNet: anything that defines the building
# shell (ceiling/floor/wall/beam/column/window/door) is PERMANENT; furniture
# and "clutter" (S3DIS's catch-all for movable stuff) is TRANSIENT.
#
# Notable S3DIS quirks vs. ScanNet:
#  - "ceiling" is a first-class label here (ScanNet doesn't separate it)
#  - "beam" and "column" exist as structural-only classes — PERMANENT
#  - No plumbing fixtures, no separate cabinet — built-ins fall under
#    "clutter" or "bookcase", which is why the permanent/transient split
#    on S3DIS is a strictly architectural cut
S3DIS_13_NAMES: Dict[int, str] = {
    0:  "ceiling",
    1:  "floor",
    2:  "wall",
    3:  "beam",
    4:  "column",
    5:  "window",
    6:  "door",
    7:  "table",
    8:  "chair",
    9:  "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter",
}

S3DIS_13_BUCKET: Dict[int, Bucket] = {
    0:  Bucket.PERMANENT,   # ceiling — building shell
    1:  Bucket.PERMANENT,   # floor   — building shell
    2:  Bucket.PERMANENT,   # wall    — building shell
    3:  Bucket.PERMANENT,   # beam    — structural
    4:  Bucket.PERMANENT,   # column  — structural
    5:  Bucket.PERMANENT,   # window  — envelope
    6:  Bucket.PERMANENT,   # door    — envelope
    7:  Bucket.TRANSIENT,   # table
    8:  Bucket.TRANSIENT,   # chair
    9:  Bucket.TRANSIENT,   # sofa
    10: Bucket.TRANSIENT,   # bookcase — usually free-standing in S3DIS rooms
    11: Bucket.TRANSIENT,   # board    — whiteboards / pinboards, removable
    12: Bucket.TRANSIENT,   # clutter  — S3DIS catch-all for movable
}

# 7 permanent / 6 transient classes for S3DIS-13.
# Palette tuned to match the ScanNet palette where labels overlap (floor,
# wall, door) for visual continuity in the Day-1-vs-Day-2 comparison.
S3DIS_13_PALETTE: Dict[int, tuple] = {
    0:  (174, 199, 232),  # ceiling — light blue (same tone as wall in ScanNet)
    1:  (152, 223, 138),  # floor   — light green (matches ScanNet floor)
    2:  (197, 197, 197),  # wall    — light grey
    3:  (255, 187, 120),  # beam    — light orange
    4:  ( 88, 130, 173),  # column  — slate blue
    5:  (197, 176, 213),  # window  — lavender (matches ScanNet window)
    6:  (214,  39,  40),  # door    — strong red (matches ScanNet door)
    7:  (255, 152, 150),  # table   — pink
    8:  (188, 189,  34),  # chair   — olive
    9:  (140,  86,  75),  # sofa    — brown (matches ScanNet sofa)
    10: (148, 103, 189),  # bookcase — purple
    11: (247, 182, 210),  # board   — light pink
    12: ( 82,  84, 163),  # clutter — navy (matches ScanNet otherfurniture)
}


# Distinguishable palette for the semantic PLY. ScanNet's official palette is
# fine but a few colors collide visually — these are tuned for a 3-pane viewer.
SCANNET_20_PALETTE: Dict[int, tuple] = {
    0:  (  0,   0,   0),  # unlabeled — black
    1:  (174, 199, 232),  # wall      — light blue
    2:  (152, 223, 138),  # floor     — light green
    3:  ( 31, 119, 180),  # cabinet
    4:  (255, 187, 120),  # bed
    5:  (188, 189,  34),  # chair
    6:  (140,  86,  75),  # sofa
    7:  (255, 152, 150),  # table
    8:  (214,  39,  40),  # door      — strong red so doors pop
    9:  (197, 176, 213),  # window
    10: (148, 103, 189),  # bookshelf
    11: (196, 156, 148),  # picture
    12: ( 23, 190, 207),  # counter
    13: (247, 182, 210),  # desk
    14: (219, 219, 141),  # curtain
    15: (255, 127,  14),  # refrigerator
    16: (158, 218, 229),  # shower curtain
    17: ( 44, 160,  44),  # toilet
    18: (112, 128, 144),  # sink
    19: (227, 119, 194),  # bathtub
    20: ( 82,  84, 163),  # otherfurniture
}


if __name__ == "__main__":
    import collections
    counts = collections.Counter(SCANNET_20_BUCKET.values())
    print(f"PERMANENT classes: {counts[Bucket.PERMANENT]}")
    print(f"TRANSIENT classes: {counts[Bucket.TRANSIENT]}")
    print()
    print(f"{'id':>3}  {'name':<16}  bucket")
    for cid, name in SCANNET_20_NAMES.items():
        print(f"{cid:>3}  {name:<16}  {SCANNET_20_BUCKET[cid].value}")
