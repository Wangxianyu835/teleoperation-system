"""Same-directory atomic replacement; callers close writers before commit."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile


@contextmanager
def atomic_output(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        if temporary.exists(): temporary.unlink()
