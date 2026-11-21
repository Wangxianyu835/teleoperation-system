"""Small runtime identity record; no model import or device initialization."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
from pathlib import Path
import platform
import subprocess
import sys


def runtime_metadata(repository_root=None):
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    locks = root/'environments/unified/locks/win-64'
    hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(locks.glob('*')) if path.is_file() and path.name != 'validation.json'}
    packages = {}
    for name in ('numpy','scipy','torch','torchvision','timm','dex-retargeting','nlopt',
                 'pin-pink','mmcv-full','mmdet','opencv-contrib-python','mediapipe','smplx'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    import dex_retargeting
    packages['dex-retargeting'] = dex_retargeting.__version__
    def git(*arguments):
        try:
            result = subprocess.run(['git','-c','safe.directory='+root.as_posix(),'-C',str(root),*arguments],
                                    capture_output=True,text=True,timeout=5)
        except (FileNotFoundError,subprocess.TimeoutExpired) as error:
            return None, type(error).__name__
        return (result.stdout.strip(), None) if result.returncode == 0 else (None,result.stderr.strip())
    commit, error = git('rev-parse','HEAD')
    state, state_error = git('status','--porcelain','--untracked-files=normal')
    return {'python':sys.executable,'python_version':platform.python_version(),
            'prefix':sys.prefix,'base_prefix':sys.base_prefix,'packages':packages,
            'lock_sha256':hashes,'code_commit':commit,'code_dirty':None if state is None else bool(state),
            'git_error':error or state_error,
            'dex_backend_source':str(Path(dex_retargeting.__file__).resolve())}
