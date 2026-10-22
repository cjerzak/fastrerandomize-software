"""Time the package's actual R-created JAX functions; no kernel reimplementation."""
import ctypes
import atexit
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import jax
import jaxlib
import numpy as np


def sync(value):
    return jax.block_until_ready(value)


def metadata():
    return {
        "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": np.__version__, "devices": [str(x) for x in jax.devices()],
        "timezone": list(time.tzname),
        "x64": jax.config.jax_enable_x64,
        "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "environment": {k: os.environ.get(k) for k in (
            "XLA_FLAGS", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader"], text=True).strip(),
    }


def stats(samples):
    return {"median_ms": 1000 * statistics.median(samples),
            "min_ms": 1000 * min(samples), "max_ms": 1000 * max(samples),
            "p10_ms": 1000 * float(np.percentile(samples, 10)),
            "p90_ms": 1000 * float(np.percentile(samples, 90)),
            "samples_ms": [1000 * x for x in samples]}


def measure(name, fn, args, output_dir, repeats=30, warmups=5):
    args = tuple(args)
    start = time.perf_counter()
    compiled = fn.lower(*args).compile()
    compile_seconds = time.perf_counter() - start
    Path(output_dir, name + ".hlo.txt").write_text(compiled.as_text())
    for _ in range(int(warmups)):
        sync(fn(*args))
    samples = []
    measurement_start_epoch = time.time()
    for _ in range(int(repeats)):
        start = time.perf_counter()
        result = sync(fn(*args))
        samples.append(time.perf_counter() - start)
    out = dict(name=name, compile_seconds=compile_seconds,
               measurement_start_epoch=measurement_start_epoch,
               measurement_end_epoch=time.time(),
               cost_analysis=compiled.cost_analysis(), **stats(samples))
    Path(output_dir, name + ".json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("samples_ms", "cost_analysis")}), flush=True)
    return result


def profile(name, fn, args, output_dir, repeats=3):
    args = tuple(args)
    for _ in range(5):
        sync(fn(*args))
    lib = ctypes.CDLL(str(Path(__file__).parent / "markers.so"))
    lib.sol_push.argtypes = [ctypes.c_char_p]
    measurement_start_epoch = time.time()
    lib.sol_start()
    for i in range(int(repeats)):
        lib.sol_push(f"{name}_{i}".encode())
        sync(fn(*args))
        lib.sol_pop()
    lib.sol_stop()
    Path(output_dir, name + "_capture.json").write_text(json.dumps({
        "name": name, "repeats": int(repeats),
        "measurement_start_epoch": measurement_start_epoch,
        "measurement_end_epoch": time.time(),
    }, indent=2))


def save_metadata(output_dir):
    Path(output_dir, "metadata.json").write_text(json.dumps(metadata(), indent=2))
    log = open(Path(output_dir, "telemetry.csv"), "w")
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
        "--format=csv", "-lms", "100"], stdout=log, stderr=subprocess.DEVNULL)
    def close():
        monitor.terminate()
        monitor.wait()
        log.close()
    atexit.register(close)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def indices(n):
    return np.unique(np.linspace(0, int(n) - 1, min(64, int(n)), dtype=np.int32))
