"""Record the measured source revision and verify the remote package snapshot."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("revision", choices=("baseline", "optimized"))
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--commit", required=True)
parser.add_argument("--remote-root", required=True)
parser.add_argument("--host", default="cjerzak-rog")
args = parser.parse_args()
files = subprocess.check_output([
    "git", "-C", str(args.source_root), "ls-tree", "-r", "--name-only", args.commit, "fastrerandomize"
], text=True).splitlines()
hashes = {str(Path(name).relative_to("fastrerandomize")):
          hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() for name in files}
code = (
    "import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
    "names=json.load(sys.stdin); print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}))"
)
remote = subprocess.run(["ssh", args.host, "python3 -c " + shlex.quote(code) + " " + shlex.quote(args.remote_root + "/source/fastrerandomize")],
                        input=json.dumps(list(hashes)), text=True, capture_output=True, check=True)
assert json.loads(remote.stdout) == hashes, "Remote source does not match the local package"
root = Path(__file__).resolve().parent
harness = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
           for name in ("benchmark.R", "timing.py", "run.sh", "calibrate.cu", "markers.cu")}
remote_harness = subprocess.run([
    "ssh", args.host, "python3 -c " + shlex.quote(code) + " " + shlex.quote(args.remote_root + "/benchmarks/speed_of_light")
], input=json.dumps(list(harness)), text=True, capture_output=True, check=True)
assert json.loads(remote_harness.stdout) == harness, "Remote measurement harness does not match"
manifest = {
    "source_commit": args.commit, "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_hashes": hashes, "remote_source_hashes_match": True,
    "host": args.host, "remote_root": args.remote_root,
    "measurement_harness_hashes": harness, "remote_harness_hashes_match": True,
    "analysis_script_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ("analyze.py", "compare.py", "plot.py")},
}
out = root / "results" / args.revision
out.mkdir(parents=True, exist_ok=True)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"{args.revision}: {len(hashes)} source files verified against {args.host}")
