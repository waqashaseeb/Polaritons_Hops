"""Create a NEW PRIVATE repository under the authenticated owner and publish this source bundle.

Run on the Mac after gh auth login, or use a connected GitHub integration.
No existing repository is overwritten and no force-push is performed.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="polariton-hops-hpc")
    parser.add_argument("--owner", default="waqashaseeb")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.repo): raise ValueError("Use a simple new repository name")
    if not shutil.which("git") or not shutil.which("gh"):
        raise RuntimeError("Git and GitHub CLI are required on the Mac; the HPC simulation itself does not require Git")
    user = json.loads(run(["gh", "api", "user"]))
    if user.get("login", "").lower()!=args.owner.lower():
        raise RuntimeError("Authenticate the requested owner account before publishing: "+args.owner)
    target = args.owner+"/"+args.repo
    existing = subprocess.run(["gh", "repo", "view", target, "--json", "nameWithOwner"], cwd=ROOT, capture_output=True, text=True)
    if existing.returncode==0:
        raise RuntimeError("Repository already exists. Choose a new name with --repo; existing content is never overwritten")
    if not (ROOT/".git").is_dir(): run(["git", "init", "-b", "main"])
    run(["git", "add", "--", "."])
    status = run(["git", "status", "--porcelain"])
    if status:
        name = user.get("name") or user["login"]
        email = str(user["id"])+"+"+user["login"]+"@users.noreply.github.com"
        run(["git", "-c", "user.name="+name, "-c", "user.email="+email, "commit", "-m", "Add reproducible polariton HOPS Slurm workflow"])
    remotes = run(["git", "remote"]).splitlines()
    if "origin" in remotes:
        raise RuntimeError("This local source already has an origin. Review it before creating or pushing a new repository")
    run(["gh", "repo", "create", target, "--private", "--description", "Conditional polariton HOPS versus Pauli studies with resumable Slurm execution"])
    url = "https://github.com/"+target+".git"
    run(["git", "remote", "add", "origin", url])
    run(["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential", "push", "-u", "origin", "main"])
    print("PUBLISHED PRIVATE REPOSITORY: https://github.com/"+target)


if __name__=="__main__": main()
