"""Summarize synchronized timings and CUDA kernel activity from Nsight Systems.

These are analytic roofline estimates, not Nsight Compute SOL counters.
Run locally after copying remote results to this directory's results/.
"""
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_json(path):
    return json.loads(path.read_text())


def profile_summary(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT k.start, k.end, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start
    """).fetchall()
    ranges = conn.execute("""
        SELECT start, end, text FROM NVTX_EVENTS
        WHERE text LIKE 'n4096_p2048%' AND end IS NOT NULL ORDER BY start
    """).fetchall()
    outputs = []
    for interval in ranges:
        kernels = [r for r in rows if r["start"] >= interval["start"] and r["end"] <= interval["end"]]
        groups = collections.defaultdict(float)
        gemms = []
        for k in kernels:
            name = k["name"]
            duration_ms = (k["end"] - k["start"]) / 1e6
            if "DeviceSegmentedRadixSortKernel" in name:
                group = "assignment_radix_sort"
            elif "sgemm" in name:
                group = "balance_gemm"
                gemms.append(duration_ms)
            elif "RadixSort" in name or "sort" in name.lower():
                group = "other_sort"
            else:
                group = "other_kernels"
            groups[group] += duration_ms
        merged = []
        for k in kernels:
            if merged and k["start"] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], k["end"])
            else:
                merged.append([k["start"], k["end"]])
        gpu_busy_ms = sum(b - a for a, b in merged) / 1e6
        outputs.append({
            "range": interval["text"],
            "wall_ms": (interval["end"] - interval["start"]) / 1e6,
            "kernel_sum_ms": sum(groups.values()), "kernel_union_ms": gpu_busy_ms,
            "kernels": len(kernels), "groups_ms": dict(groups),
            "gemm_count": len(gemms), "gemm_samples_ms": gemms,
            "gemm_median_ms": statistics.median(gemms) if gemms else None,
        })
    return outputs


def main():
    b, n, p = 4096, 4096, 2048
    work = 2 * b * n * p
    bytes_gemm = 4 * (b*n + n*p + b*p)
    # NVIDIA's listed upper boost clock is a reference, not a locked clock.
    fp32_reference = 7680 * 2 * 2287e6
    # CUDA reports 896.064 GB/s on this machine (28.002 Gb/s, 256-bit bus).
    bandwidth_reference = 896.064e9
    summary = {
        "shape": {"batch": b, "units": n, "covariates": p},
        "balance_gemm_flops": work, "gemm_minimum_bytes": bytes_gemm,
        "gemm_arithmetic_intensity": work / bytes_gemm,
        "reference_fp32_flops_per_second": fp32_reference,
        "reference_dram_bytes_per_second": bandwidth_reference,
        "gemm_compute_lower_bound_ms": work / fp32_reference * 1000,
        "gemm_dram_lower_bound_ms": bytes_gemm / bandwidth_reference * 1000,
        "calibration": [json.loads(line) for line in (RESULTS / "calibration.jsonl").read_text().splitlines()],
        "profiles": {}, "sweep": [],
    }
    for path in sorted(RESULTS.glob("profile_*.sqlite")):
        summary["profiles"][path.stem] = profile_summary(path)
    for path in sorted((RESULTS / "sweep").glob("*.json")):
        match = re.fullmatch(r"n(\d+)_p(\d+)_b(\d+)_(balance|batch)", path.stem)
        if not match:
            continue
        nn, pp, bb = map(int, match.groups()[:3])
        data = read_json(path)
        summary["sweep"].append({"n": nn, "p": pp, "batch": bb, "scope": match[4],
                                 "median_ms": data["median_ms"],
                                 "p10_ms": data["p10_ms"], "p90_ms": data["p90_ms"],
                                 "useful_balance_tflops": 2*nn*pp*bb / (data["median_ms"] * 1e9)})
    for mode in ("pipeline", "focus", "api", "hostprep"):
        summary[mode] = {path.stem: read_json(path) for path in sorted((RESULTS / mode).glob("n*.json"))}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    with (RESULTS / "sweep_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary["sweep"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["sweep"])
    # Normalize readable tool exports for Git without changing numeric data.
    # Original raw exports remain on the remote host.
    for path in list(RESULTS.glob("*.log")) + list(RESULTS.glob("*.txt")):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("profiles", "sweep", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
