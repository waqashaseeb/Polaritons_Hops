"""Syntax, path and input checks. Does not import scientific modules or run HOPS."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import py_compile
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
HPC_ROOT = "/home/tariq.qayyum/hpc/polariton_hops"


def validate(root):
    root = Path(root).resolve()
    for name in ("logs", "results", "outputs", "figures", "reports", "checkpoints", "runtime"):
        path = root/name
        if not path.resolve().is_relative_to(root): raise ValueError("Output directory escapes project: "+str(path))
        path.mkdir(exist_ok=True)
    cache = root/"outputs/local_checks/compiled"
    cache.mkdir(parents=True, exist_ok=True)
    errors, compiled, shell = [], [], []
    for path in sorted(root.rglob("*.py")):
        if any(part in (".venv", ".tools", ".git", "runtime", "outputs") for part in path.relative_to(root).parts): continue
        try:
            relative = path.relative_to(root)
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative), feature_version=(3,11))
            target = cache/relative.with_suffix(".pyc")
            target.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(str(path), cfile=str(target), doraise=True)
            compiled.append(str(relative))
        except (py_compile.PyCompileError, SyntaxError) as exc: errors.append(str(exc))
    scripts = list(root.glob("Batch"))+list(root.rglob("*.sh"))+list(root.rglob("*.sbatch"))
    for path in sorted(scripts):
        if any(part in (".venv", ".tools", ".git", "runtime", "outputs") for part in path.relative_to(root).parts): continue
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        if result.returncode: errors.append(str(path.relative_to(root))+": "+result.stderr)
        else: shell.append(str(path.relative_to(root)))
    batch = (root/"Batch").read_text()
    checks = {"CPU_queue_longq": "#SBATCH --partition=longq" in batch,
              "forty_CPU_slots": "#SBATCH --ntasks=40" in batch,
              "one_CPU_per_task": "#SBATCH --cpus-per-task=1" in batch,
              "walltime_23_hours": "#SBATCH --time=23:00:00" in batch,
              "no_GPU_request_in_Batch": not re.search(r"^#SBATCH.*--(?:gres|gpus)", batch, re.M),
              "absolute_project_stdout": HPC_ROOT+"/logs/%x_%j.out" in batch,
              "absolute_project_stderr": HPC_ROOT+"/logs/%x_%j.err" in batch,
              "local_environment_declared": HPC_ROOT in (root/"bootstrap.sh").read_text() and "$PROJECT_ROOT/.venv/bin/python" in (root/"bootstrap.sh").read_text(),
              "no_shared_conda_mutating_command": True}
    for path in scripts:
        if path.is_file():
            text = path.read_text()
            if re.search(r"^\s*(?:conda\s+(?:install|update)\s+(?:-n|--name)\s+base|(?:pip|pip3)\s+install|.*>\s*/cm/shared/apps/anaconda3)", text, re.M):
                checks["no_shared_conda_mutating_command"] = False
                errors.append("Unsafe install command: "+str(path))
    for key, passed in checks.items():
        if not passed: errors.append("Static policy check failed: "+key)
    expected = json.loads((root/"engine/inputs/expected_backend_files.json").read_text())
    for name, wanted in expected.items():
        if hashlib.sha256((root/"engine"/name).read_bytes()).hexdigest()!=wanted:
            errors.append("Pinned input/source checksum mismatch: "+name)
    for path in (root/"configs").glob("*.json"):
        try: json.loads(path.read_text())
        except ValueError as exc: errors.append(str(path)+": "+str(exc))
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              "status": "PASS" if not errors else "FAIL", "python_files_compiled": len(compiled), "shell_files_bash_n": len(shell),
              "pinned_backend_checksums": len(expected), "checks": checks, "errors": errors,
              "real_execution": "Compilation, shell parsing, directory/path checks, JSON parsing and source/input checksum checks only",
              "HOPS_trajectories_run": 0, "HPC_submission_performed": False,
              "unverified": ["private cluster connectivity", "actual queue MaxTime and allocation policy", "availability of anaconda3/Python and package downloads", "new HOPS convergence and runtime"]}
    out = root/"docs/LOCAL_VALIDATION.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    (root/"docs/LOCAL_VALIDATION.md").write_text("# Local validation\n\nStatus: "+result["status"]+"\n\nPython files compiled: "+str(len(compiled))+"\nShell scripts parsed with bash -n: "+str(len(shell))+"\nPinned source checksums: "+str(len(expected))+"\n\nNo new HOPS propagation or HPC submission was performed. Actual queue limits and runtime remain unverified until cluster preflight/execution.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return not errors


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.project) else 1)
