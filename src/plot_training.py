"""Plot training curves from the STGNN training log."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re

LOG = r"C:\Algo-C2\training.log"

# Parse log
folds = {}
current_fold = None

with open(LOG) as f:
    for line in f:
        m = re.match(r"\s+Fold (\d+): train=(\d+), val=(\d+)", line)
        if m:
            current_fold = int(m.group(1))
            folds[current_fold] = {"train_size": int(m.group(2)),
                                   "val_size": int(m.group(3)),
                                   "epochs": [], "train_mse": [], "val_mse": [],
                                   "acc": None, "r2": None}
            continue

        m = re.match(r"\s+Epoch (\d+): train_mse=([\d.]+), val_mse=([\d.]+)", line)
        if m and current_fold is not None:
            folds[current_fold]["epochs"].append(int(m.group(1)))
            folds[current_fold]["train_mse"].append(float(m.group(2)))
            folds[current_fold]["val_mse"].append(float(m.group(3)))
            continue

        m = re.match(r"\s+Fold \d+: MSE=[\d.]+, R2=([-\d.]+), Acc=([\d.]+)%", line)
        if m and current_fold is not None:
            folds[current_fold]["r2"] = float(m.group(1))
            folds[current_fold]["acc"] = float(m.group(2))

# Colors per fold
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
fig.suptitle("STGNN Training — Old Run (Baseline)\n35 FX Pairs · 17,998 M1 Bars · 18 Days Data",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Training MSE per fold ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#161b22")
for fold_id, data in folds.items():
    if data["epochs"]:
        ax1.plot(data["epochs"], data["train_mse"],
                 color=COLORS[fold_id], linewidth=2, label=f"Fold {fold_id}")
ax1.set_title("Train MSE per Fold", color="white", fontsize=11)
ax1.set_xlabel("Epoch", color="#8b949e")
ax1.set_ylabel("MSE", color="#8b949e")
ax1.tick_params(colors="#8b949e")
ax1.spines[:].set_color("#30363d")
ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
ax1.grid(alpha=0.15, color="white")

# ── Plot 2: Validation MSE per fold ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#161b22")
for fold_id, data in folds.items():
    if data["epochs"]:
        ax2.plot(data["epochs"], data["val_mse"],
                 color=COLORS[fold_id], linewidth=2, linestyle="--",
                 label=f"Fold {fold_id}")
ax2.set_title("Validation MSE per Fold", color="white", fontsize=11)
ax2.set_xlabel("Epoch", color="#8b949e")
ax2.set_ylabel("MSE", color="#8b949e")
ax2.tick_params(colors="#8b949e")
ax2.spines[:].set_color("#30363d")
ax2.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
ax2.grid(alpha=0.15, color="white")

# ── Plot 3: Train vs Val MSE overlay (best fold = fold 1) ─────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#161b22")
for fold_id, data in folds.items():
    if data["epochs"]:
        ax3.plot(data["epochs"], data["train_mse"],
                 color=COLORS[fold_id], linewidth=1.5, alpha=0.6)
        ax3.plot(data["epochs"], data["val_mse"],
                 color=COLORS[fold_id], linewidth=1.5, linestyle=":", alpha=0.9)
# Dummy lines for legend
ax3.plot([], [], color="white", linewidth=2, label="Train (solid)")
ax3.plot([], [], color="white", linewidth=2, linestyle=":", label="Val (dotted)")
ax3.set_title("Train vs Val MSE Overlay (all folds)", color="white", fontsize=11)
ax3.set_xlabel("Epoch", color="#8b949e")
ax3.set_ylabel("MSE", color="#8b949e")
ax3.tick_params(colors="#8b949e")
ax3.spines[:].set_color("#30363d")
ax3.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
ax3.grid(alpha=0.15, color="white")

# Annotate overfitting gap
for fold_id, data in folds.items():
    if data["epochs"] and len(data["val_mse"]) > 1:
        last_ep = data["epochs"][-1]
        gap = data["val_mse"][-1] - data["train_mse"][-1]
        ax3.annotate(f"Δ{gap:.3f}", xy=(last_ep, data["val_mse"][-1]),
                     color=COLORS[fold_id], fontsize=7, ha="left")

# ── Plot 4: CV Summary Bar Chart ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#161b22")

fold_ids = sorted(folds.keys())
accs = [folds[f]["acc"] for f in fold_ids if folds[f]["acc"] is not None]
r2s  = [folds[f]["r2"]  for f in fold_ids if folds[f]["r2"]  is not None]
x = np.arange(len(fold_ids))
w = 0.35

bars1 = ax4.bar(x - w/2, accs, w, color=[COLORS[i] for i in range(len(accs))],
                alpha=0.85, label="Accuracy %")
ax4.axhline(50, color="white", linestyle="--", linewidth=1, alpha=0.4, label="50% line")
ax4.axhline(33.3, color="#ff5722", linestyle="--", linewidth=1, alpha=0.4, label="Random (33%)")

# R² on secondary axis
ax4b = ax4.twinx()
ax4b.plot(x, r2s, color="#FFD700", marker="o", linewidth=2, markersize=6, label="R²")
ax4b.set_ylabel("R²", color="#FFD700")
ax4b.tick_params(axis="y", colors="#FFD700")
ax4b.set_ylim(-0.01, 0.02)
ax4b.spines[:].set_color("#30363d")

for bar, acc in zip(bars1, accs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{acc:.1f}%", ha="center", va="bottom", color="white", fontsize=8)

ax4.set_title("CV Results: Accuracy & R² per Fold", color="white", fontsize=11)
ax4.set_xlabel("Fold", color="#8b949e")
ax4.set_ylabel("Accuracy (%)", color="#8b949e")
ax4.set_xticks(x)
ax4.set_xticklabels([f"F{i}" for i in fold_ids])
ax4.tick_params(colors="#8b949e")
ax4.spines[:].set_color("#30363d")
ax4.set_ylim(40, 60)
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
           facecolor="#161b22", labelcolor="white")
ax4.grid(alpha=0.15, color="white", axis="y")

out = r"C:\Algo-C2\training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
