"""Aggregate eval/results_epoch{00,01,02}.json into a single eval/results.json
(with all three for trajectory + the winner highlighted) and emit
eval/confusion.png for the winner.

Run: python src/eval_summary.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
CKPT_DIR = ROOT / "checkpoints" / "finetune"


def load_epoch_results(epoch_str: str) -> dict:
    p = EVAL_DIR / f"results_epoch{epoch_str}.json"
    return json.loads(p.read_text())


def main():
    epochs = ["00", "01", "02"]
    runs = {e: load_epoch_results(e) for e in epochs}

    # Pick winner by binary F1 (sem->bucket, the headline)
    by_f1 = {e: runs[e]["binary_sem"]["f1"] for e in epochs}
    winner = max(by_f1, key=by_f1.get)
    runs_winner = runs[winner]

    summary = dict(
        all_epochs={
            e: dict(
                mIoU=runs[e]["mIoU"],
                overall_acc=runs[e]["overall_acc"],
                binary_sem=runs[e]["binary_sem"],
                binary_pt=runs[e]["binary_pt"],
            )
            for e in epochs
        },
        winner_epoch=winner,
        winner_ckpt=f"checkpoints/finetune/ptv3_pt_epoch{winner}.pth",
        winner_metrics=dict(
            mIoU=runs_winner["mIoU"],
            overall_acc=runs_winner["overall_acc"],
            binary_sem=runs_winner["binary_sem"],
            binary_pt=runs_winner["binary_pt"],
            per_class=runs_winner["per_class"],
        ),
    )

    out = EVAL_DIR / "results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(f"\nWinner: epoch {winner}  binary F1 = {by_f1[winner]*100:.2f}%")

    # ---- confusion matrix plot for the winner ----
    # Build per-class confusion (S3DIS-13 × S3DIS-13) is too noisy; produce
    # the 2x2 perm/transient one — the headline visual for the README.
    cm_sem = np.asarray(runs_winner["confusion_sem"], dtype=np.int64)
    cm_pt = np.asarray(runs_winner["confusion_pt"], dtype=np.int64)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, cm, title in (
        (axes[0], cm_sem, "Sem→bucket (S3DIS-13 argmax → PT bucket)"),
        (axes[1], cm_pt,  "PT head (direct 2-class)"),
    ):
        cm_pct = cm / cm.sum() * 100.0
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred transient", "pred permanent"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["gt transient", "gt permanent"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{cm_pct[i, j]:.1f}%\n({cm[i, j]:,})",
                        ha="center", va="center",
                        color="white" if cm_pct[i, j] > 50 else "black",
                        fontsize=9)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("ground truth", fontsize=9)

    fig.suptitle(
        f"S3DIS Area-5 binary permanent/transient confusion — "
        f"PTv3 epoch {winner} (mIoU={runs_winner['mIoU']*100:.1f}%, "
        f"F1={runs_winner['binary_sem']['f1']*100:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = EVAL_DIR / "confusion.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")

    # Console table of per-class IoU (for README + log)
    print(f"\n=== Per-class IoU (epoch {winner}) ===")
    print(f"{'cls':<10} {'IoU':>7}  {'bucket':<10}")
    for r in runs_winner["per_class"]:
        iou = r["iou"]
        iou_s = f"{iou*100:6.1f}%" if iou is not None and not (isinstance(iou, float) and iou != iou) else "  nan%"
        print(f"  {r['name']:<10} {iou_s}")


if __name__ == "__main__":
    main()
