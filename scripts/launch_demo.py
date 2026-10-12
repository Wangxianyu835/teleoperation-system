"""Launch the isolated Allegro demo with project DLL paths and temporary directories."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / ".conda-env"


def runtime_env():
    env = os.environ.copy()
    windows = Path(os.environ.get("SystemRoot", "C:/Windows"))
    locations = [PREFIX, PREFIX / "Library/mingw-w64/bin", PREFIX / "Library/usr/bin",
                 PREFIX / "Library/bin", PREFIX / "Scripts", PREFIX / "bin",
                 windows / "System32", windows]
    env["PATH"] = os.pathsep.join(map(str, locations))
    env["CONDA_PREFIX"] = str(PREFIX)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
    for name, relative in {"TEMP": ".tmp", "TMP": ".tmp",
                           "MPLCONFIGDIR": ".cache/matplotlib", "IPYTHONDIR": ".cache/ipython",
                           "JUPYTER_CONFIG_DIR": ".cache/jupyter",
                           "PYTHONUSERBASE": ".cache/python-user"}.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def main():
    python = PREFIX / "python.exe"
    if not python.is_file():
        raise RuntimeError("Project environment is missing; complete installation first")
    command = [str(python), "-I", "-B", str(ROOT / "run_windows_allegro.py"), *sys.argv[1:]]
    try:
        returncode = subprocess.call(command, cwd=ROOT, env=runtime_env())
    except KeyboardInterrupt:
        returncode = 130
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
