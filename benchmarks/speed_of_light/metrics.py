"""Derive report metrics from the recorded summary and GPU telemetry."""
import argparse
import json
import csv
import datetime
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def profile_clocks(directory):
    capture = json.loads(next(directory.glob("*_capture.json")).read_text())
    rows = []
    with (directory / "telemetry.csv").open() as stream:
        for row in csv.DictReader(stream):
            row = {k.strip(): v.strip() for k, v in row.items()}
            epoch = datetime.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=ZoneInfo("America/Chicago")).timestamp()
            if (capture["measurement_start_epoch"] <= epoch <= capture["measurement_end_epoch"]
                    and float(row["utilization.gpu [%]"].split()[0]) >= 90):
                rows.append(row)
    clocks = [float(row["clocks.current.sm [MHz]"].split()[0]) for row in rows]
    return {"samples_at_least_90_percent_gpu_utilization": len(clocks),
            "sm_clock_mhz_min": min(clocks), "sm_clock_mhz_median": statistics.median(clocks),
            "sm_clock_mhz_max": max(clocks)}


def metrics(summary):
    api = next(iter(summary["api"].values()))
    prep = next(iter(summary["hostprep"].values()))
    pipeline = next(iter(summary["pipeline"].values()))
    traces = summary["profiles"]["profile_all"]
    gemms = [v for trace in traces for v in trace["gemm_samples_ms"]]
    gemm_ms = statistics.median(gemms)
    groups = {name: statistics.mean(t["groups_ms"].get(name, 0) for t in traces)
              for name in sorted({k for t in traces for k in t["groups_ms"]})}
    total = sum(groups.values())
    tflops = summary["balance_gemm_flops"] / (gemm_ms * 1e9)
    return {
        "api_median_seconds": api["median_seconds"],
        "api_range_seconds": [min(api["samples_seconds"]), max(api["samples_seconds"])],
        "first_api_call_after_backend_setup_seconds": api["first_api_call_after_backend_setup_seconds"],
        "preparation_median_seconds": prep["median_seconds"],
        "preparation_range_seconds": [min(prep["samples_seconds"]), max(prep["samples_seconds"])],
        "pipeline_median_seconds": pipeline["median_ms"] / 1000,
        "pipeline_p10_p90_seconds": [pipeline["p10_ms"] / 1000, pipeline["p90_ms"] / 1000],
        "gemm_samples": len(gemms), "gemm_median_ms": gemm_ms,
        "gemm_tflops": tflops,
        "gemm_fraction_of_upper_boost_reference": tflops * 1e12 / summary["reference_fp32_flops_per_second"],
        "kernel_groups_mean_ms": groups,
        "kernel_groups_fraction": {k: v / total for k, v in groups.items()},
        "kernel_busy_fraction": statistics.mean(t["kernel_union_ms"] / t["wall_ms"] for t in traces),
        "max_relative_error_float64": api["max_relative_error_float64"],
        "accepted": api["accepted"],
        "repeat_identical": api["repeat_identical"],
        "diagonal_reference_identical": {
            "Xw": prep["identical_Xw"],
            "inverse_variance": prep["identical_inverse_variance"],
        },
    }


def main(results=RESULTS):
    summary = json.loads((results / "summary.json").read_text())
    measured = metrics(summary)
    measured["profile_telemetry"] = profile_clocks(results / "profile_all")
    (results / "metrics.json").write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    main(parser.parse_args().results_dir)
