"""Standard-library-only interpreter and process environment preparation."""
from __future__ import annotations

import os
import ntpath
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_VARS = (
    ('PIP_CACHE_DIR', 'pip'), ('MPLCONFIGDIR', 'matplotlib'),
    ('HF_HOME', 'huggingface'), ('TORCH_HOME', 'torch'),
    ('XDG_CACHE_HOME', 'xdg'), ('TEMP', 'tmp'), ('TMP', 'tmp'),
)


def cache_environment(repository_root=None, *, environ=None, create=False):
    """Keep explicit non-C caches; supply project defaults without global changes.

    Only create the defaults this project selects. Existing custom cache paths
    remain the caller's responsibility, and unrelated user caches are untouched.
    """
    env = dict(os.environ if environ is None else environ)
    cache_root = Path(repository_root or PROJECT_ROOT)/'outputs/cache'
    for name, subdirectory in CACHE_VARS:
        current = env.get(name, '')
        if current and ntpath.splitdrive(current)[0].lower() != 'c:':
            continue
        target = cache_root/subdirectory
        if create:
            target.mkdir(parents=True, exist_ok=True)
        env[name] = str(target)
    return env


def resolve_project_python(repository_root, explicit_python=None) -> Path:
    """Resolve an explicit interpreter or the project's single Conda prefix."""
    target = Path(explicit_python) if explicit_python else Path(repository_root)/'.envs/teleoperation/python.exe'
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f'Teleoperation Python not found: {target}; run devtools/setup_unified_environment.ps1 or provide an explicit interpreter')
    return target


def interpreter_prefix(python) -> Path:
    parent = Path(python).expanduser().resolve().parent
    return parent.parent if parent.name.lower() in ('scripts', 'bin') else parent


def _dll_directories(prefix):
    return [path for path in (prefix, prefix/'Library/bin', prefix/'Library/usr/bin',
                             prefix/'Library/mingw-w64/bin', prefix/'DLLs') if path.is_dir()]


def child_environment(python, *, environ=None):
    """Build a process-local environment without inherited Python/Conda paths."""
    env = cache_environment(environ=environ, create=True)
    prefix = interpreter_prefix(python)
    old_prefixes = [Path(value).resolve() for key, value in env.items()
                    if (key.upper().startswith('CONDA_PREFIX') or key.upper() in ('VIRTUAL_ENV', 'PYTHONHOME')) and value]
    for key in list(env):
        if key.upper() in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONSTARTUP', 'VIRTUAL_ENV') or key.upper().startswith('CONDA_'):
            del env[key]
    paths = []
    for value in env.get('PATH', '').split(os.pathsep):
        if not value:
            continue
        path = Path(value).resolve()
        if path == prefix or prefix in path.parents:
            continue  # selected prefix is inserted once, in a known order
        if any(path == old or old in path.parents for old in old_prefixes):
            continue
        if any((ancestor/'conda-meta').is_dir() or (ancestor/'pyvenv.cfg').is_file()
               for ancestor in (path, *path.parents)):
            continue
        if any(part.lower() in ('site-packages', 'outputs', '.venv', '.envs') for part in path.parts):
            continue
        paths.append(value)
    env['PATH'] = os.pathsep.join([str(path) for path in _dll_directories(prefix)] + paths)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def initialize_windows_dlls(prefix=None, *, platform=None):
    """Return handles the caller must retain for its native import lifetime."""
    if (os.name if platform is None else platform) != 'nt':
        return []
    return [os.add_dll_directory(str(path)) for path in _dll_directories(Path(prefix or sys.prefix))]


def python_script_command(python, script, arguments=()):
    """Initialize the selected prefix before executing a worker's native imports."""
    src = str(Path(__file__).resolve().parents[1])
    bootstrap = (
        'import sys,runpy;'
        f'sys.path.insert(0,{src!r});'
        'from teleoperation.runtime_environment import initialize_windows_dlls;'
        '_teleop_dll_handles=initialize_windows_dlls();'
        'script=sys.argv.pop(1);sys.argv[0]=script;'
        'from pathlib import Path;sys.path.insert(0,str(Path(script).parent));'
        'runpy.run_path(script,run_name="__main__")'
    )
    return [str(python), '-E', '-s', '-c', bootstrap, str(script), *map(str, arguments)]


def _main():
    """PowerShell boundary: use this file before importing native dependencies."""
    import runpy
    args = sys.argv[1:]
    if args and args[0] == '--resolve':
        print(resolve_project_python(args[1], args[2] if len(args) > 2 else None))
        return
    env = child_environment(sys.executable)
    os.environ.clear()
    os.environ.update(env)
    # This entry also works before the editable project install, e.g. preflight.
    selected_source = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, selected_source)
    os.environ['PYTHONPATH'] = selected_source
    dll_handles = initialize_windows_dlls()
    if not args:
        raise SystemExit('Expected -m module, -c command, or script path')
    if args[0] in ('-m', '-c'):
        # Match python -m/-c: the caller's working directory is importable.
        sys.path.insert(0, str(Path.cwd()))
    if args[0] == '-m':
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name='__main__', alter_sys=True)
    elif args[0] == '-c':
        sys.argv = ['-c', *args[2:]]
        exec(compile(args[1], '<string>', 'exec'), {'__name__': '__main__'})
    else:
        sys.argv = args
        sys.path.insert(0, str(Path(args[0]).resolve().parent))
        runpy.run_path(args[0], run_name='__main__')


if __name__ == '__main__':
    _main()
