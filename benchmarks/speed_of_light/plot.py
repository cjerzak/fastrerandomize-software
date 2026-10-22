"""Generate the report figure from recorded results, without editing images."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
s = json.loads((root / "results/optimized/summary.json").read_text())
comparison = json.loads((root / "results/comparison.json").read_text())
before, after = comparison["baseline"], comparison["optimized"]
traces = s["profiles"]["profile_all"]
gemms = [ms for t in traces for ms in t["gemm_samples_ms"]]
gemm_tflops = s["balance_gemm_flops"] / (statistics.median(gemms) * 1e9)
api = next(iter(s["api"].values()))
pipeline = next(iter(s["pipeline"].values()))
palette = {"gemm": "#167d9a", "sort": "#d88828", "other": "#9ba8b5", "host": "#536aa2"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), gridspec_kw={"width_ratios": [1.25, 1, 1.2]})

ax = axes[0]
x = np.geomspace(.2, 2048, 400)
peak = s["reference_fp32_flops_per_second"] / 1e12
bw = s["reference_dram_bytes_per_second"] / 1e12
ax.loglog(x, np.minimum(x*bw, peak), color="#374151", linewidth=2)
ax.scatter([s["gemm_arithmetic_intensity"]], [gemm_tflops], s=65, color=palette["gemm"], zorder=4)
ax.annotate(f"Observed GEMM\n{gemm_tflops:.1f} TFLOP/s", (512, gemm_tflops),
            xytext=(105, 3.7), arrowprops={"arrowstyle": "-", "color": palette["gemm"]})
ax.text(1.1, 56, f"FP32 reference: {peak:.1f} TFLOP/s", fontsize=9)
ax.text(.42, .32, "DRAM reference: 896 GB/s", rotation=37, fontsize=8)
ax.set(xlim=(.2, 2048), ylim=(.15, 100), xlabel="GEMM arithmetic intensity (FLOP/byte)",
       ylabel="Throughput (TFLOP/s)", title="Balance matrix multiplication")
ax.grid(alpha=.15, which="major")

ax = axes[1]
groups = {
    "Permutation sorting": statistics.mean(t["groups_ms"]["assignment_radix_sort"] for t in traces)/256,
    "Balance GEMM": statistics.mean(t["groups_ms"]["balance_gemm"] for t in traces)/256,
    "Other kernels": statistics.mean(t["groups_ms"]["other_kernels"] + t["groups_ms"].get("other_sort", 0)
                                       for t in traces)/256,
}
left = 0
for (name, value), color in zip(groups.items(), [palette["sort"], palette["gemm"], palette["other"]]):
    ax.barh(0, value, left=left, color=color, height=.43, label=name)
    if value > 1:
        ax.text(left+value/2, 0, f"{value/sum(groups.values()):.0%}", ha="center", va="center", color="white", weight="bold")
    left += value
ax.set(yticks=[], ylim=(-.65, .65), xlim=(0, max(12.5, left * 1.06)), xlabel="GPU kernel time per 4,096 draws (ms)",
       title="Where GPU time goes")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.02, -.02), fontsize=9)
ax.text(0, .45, f"1.05 million draws: {after['pipeline_median_seconds']:.2f} s unprofiled", fontsize=9)

ax = axes[2]
names = ["Baseline R API", "Patched R API", "Baseline preparation", "Patched preparation", "Patched GPU stage"]
values = [before["api_median_seconds"], after["api_median_seconds"],
          before["preparation_median_seconds"], after["preparation_median_seconds"],
          after["pipeline_median_seconds"]]
colors = ["#9ba8b5", "#374151", "#aab4cf", palette["host"], palette["gemm"]]
for i, (name, value, color) in enumerate(zip(names, values, colors)):
    ax.barh(i, value, color=color, height=.5)
    label = f"{value:.3f} s" if value < 1 else f"{value:.2f} s"
    ax.text(value+.25, i, label, va="center", fontsize=9)
ax.set(yticks=range(5), yticklabels=names, xlim=(0, max(values) * 1.2), xlabel="Elapsed time (s)",
       title=f"Complete R call: {comparison['api_speedup']:.2f}× faster")
ax.invert_yaxis()
ax.tick_params(axis="y", length=0, labelsize=9)
fig.suptitle("Speed of Light analysis of Monte Carlo rerandomization", y=1.02, fontsize=15, weight="bold")
fig.text(.02, -.035,
         "RTX 5080 Laptop GPU · 4,096 units · 2,048 covariates · FP32 HIGHEST · reference roofline uses 2,287 MHz, not locked clocks.\n"
         "Baseline and patched outputs are identical. Preparation and GPU-stage times were measured separately from the complete R call.",
         fontsize=9, color="#4b5563")
fig.tight_layout(w_pad=2.5)
for ext in ("png", "pdf", "svg"):
    fig.savefig(root / f"speed_of_light.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
svg = root / "speed_of_light.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
