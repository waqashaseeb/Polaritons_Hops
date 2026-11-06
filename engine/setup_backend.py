"""Copy the user's already-used pinned MesoHOPS core into this new package.

No old file is edited and no package is downloaded or upgraded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import types
from importlib.machinery import ModuleSpec

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(existing_dir):
    expected = json.loads((ROOT/'inputs/expected_backend_files.json').read_text(encoding='utf-8'))
    if all((ROOT/name).is_file() and sha(ROOT/name) == digest for name, digest in expected.items()):
        return {'status': 'PINNED_BACKEND_ALREADY_PRESENT', 'files': expected}
    old = Path(existing_dir).expanduser().resolve()
    if old == ROOT:
        raise ValueError('The existing workflow directory must be the old depth-sampling package')
    missing = [name for name, digest in expected.items() if not (old/name).is_file() or sha(old/name) != digest]
    if missing:
        raise FileNotFoundError('Pinned backend files are missing or changed under '+str(old)+
                                '. Set existing_workflow_dir in configs/pilot.json to the original polariton_depth_sampling_v1 folder. '
                                'Do not install a newer mesohops as a substitute. First missing/changed files: '+', '.join(missing[:4]))
    for name in expected:
        dest = ROOT/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old/name, dest)
    if any(sha(ROOT/name) != digest for name, digest in expected.items()):
        raise ValueError('Copied backend hash verification failed')
    return {'status': 'PINNED_BACKEND_COPIED_AND_HASH_VERIFIED', 'source_directory': str(old), 'files': expected}


def activate():
    """Bind the legacy namespace package to the verified local source.

    A regular site-packages installation can otherwise override a namespace
    directory even when its parent is first on sys.path. No source is edited.
    """
    core = (ROOT/'third_party/mesohops').resolve()
    if not (core/'dynamics/hops_trajectory.py').is_file():
        raise FileNotFoundError('Run prepare to copy the pinned backend first')
    existing = sys.modules.get('mesohops')
    if existing is not None:
        paths = [Path(p).resolve() for p in getattr(existing, '__path__', [])]
        if paths != [core]:
            raise RuntimeError('Another mesohops is already imported; use the supplied fresh-process launcher')
        return
    module = types.ModuleType('mesohops')
    module.__path__ = [str(core)]
    module.__package__ = 'mesohops'
    spec = ModuleSpec('mesohops', loader=None, is_package=True)
    spec.submodule_search_locations = [str(core)]
    module.__spec__ = spec
    sys.modules['mesohops'] = module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-dir', type=Path, required=True)
    print(json.dumps(setup(parser.parse_args().existing_dir), indent=2))
