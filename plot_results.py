"""
Regenerate the results charts for the mild-severity (PHQ-8 5-9) band.

Reads only aggregate metrics from model_results_mild_band.csv (balanced accuracy,
sensitivity, specificity averaged over 15 runs). Contains NO participant data,
transcripts, or PHQ-8 scores -- it is safe to share publicly.

Usage:
    pip install pandas matplotlib
    python plot_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt

ACCENT = "#1f4e5f"
HIGHLIGHT = "#c0392b"
CHANCE = 0.50

df = pd.read_csv("model_results_mild_band.csv")
classical = df[df["family"] == "Classical"].reset_index(drop=True)

# ---- Chart 1: balanced accuracy by classical model (with error bars) ----
fig, ax = plt.subplots(figsize=(9, 5))
colors = [HIGHLIGHT if m == "Logistic Regression" else ACCENT for m in classical["model"]]
ax.bar(classical["model"], classical["balanced_accuracy"],
       yerr=classical["ba_std"], capsize=4, color=colors, edgecolor="white")
ax.axhline(CHANCE, ls="--", lw=1.2, color="gray")
ax.text(len(classical) - 0.5, CHANCE + 0.005, "chance (0.50)", color="gray",
        ha="right", va="bottom", fontsize=9)
ax.set_ylim(0, 0.8)
ax.set_ylabel("Balanced accuracy (15-run mean)")
ax.set_title("Classical model performance — mild band (PHQ-8 5–9)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("balanced_accuracy_by_model.png", dpi=150)
plt.close()

# ---- Chart 2: best classical vs BERT (the headline finding) ----
compare = pd.DataFrame({
    "label": ["Logistic Regression\n(best classical)", "BERT\n(lr 1e-6)", "BERT\n(lr 5e-6)"],
    "ba": [0.63, 0.48, 0.45],
    "err": [0.09, 0.03, 0.04],
})
fig, ax = plt.subplots(figsize=(7, 5))
bar_colors = [HIGHLIGHT, "#7f8c8d", "#95a5a6"]
ax.bar(compare["label"], compare["ba"], yerr=compare["err"], capsize=5,
       color=bar_colors, edgecolor="white")
ax.axhline(CHANCE, ls="--", lw=1.2, color="gray")
ax.text(2.4, CHANCE + 0.005, "chance (0.50)", color="gray", ha="right",
        va="bottom", fontsize=9)
ax.set_ylim(0, 0.8)
ax.set_ylabel("Balanced accuracy")
ax.set_title("Classical baseline outperformed the fine-tuned transformer")
plt.tight_layout()
plt.savefig("classical_vs_bert.png", dpi=150)
plt.close()

print("Saved balanced_accuracy_by_model.png and classical_vs_bert.png")
