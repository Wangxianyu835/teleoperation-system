"""Lazy boundary for the official SMPLer-X adapter.

Importing this module must not import torch, OpenMMLab, or SMPL-X.  The
model-specific implementation is loaded only by the isolated worker after
asset checks pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmplerXAssets:
    source_root: Path
    checkpoint: Path
    body_model_dir: Path
    model: str = "s32"

    def body_files(self) -> dict[str, Path]:
        names = [f'smplx/SMPLX_{gender}.npz' for gender in ('NEUTRAL', 'MALE', 'FEMALE')]
        names += ['smplx/SMPLX_to_J14.pkl', 'smplx/MANO_SMPLX_vertex_ids.pkl',
                  'smplx/SMPL-X__FLAME_vertex_ids.npy']
        files = {name: self.body_model_dir / name for name in names}
        for gender, short in (('NEUTRAL', 'neutral'), ('MALE', 'm'), ('FEMALE', 'f')):
            name = f'smpl/SMPL_{gender}.pkl'
            original = self.body_model_dir / name
            files[name] = (original if original.is_file() else self.body_model_dir / 'smpl' /
                           f'basicmodel_{short}_lbs_10_207_0_v1.1.0.pkl')
        return files

    def validate(self) -> None:
        if self.model != "s32":
            raise ValueError(f"unsupported SMPLer-X model: {self.model!r}")
        for name, path in (("source_root", self.source_root), ("checkpoint", self.checkpoint),
                           ("body_model_dir", self.body_model_dir)):
            if not path.exists():
                raise FileNotFoundError(f"SMPLer-X {name} not found: {path}")
        for name, path in self.body_files().items():
            if not path.is_file():
                raise FileNotFoundError(f'SMPLer-X body asset {name} not found: {path}')

    def prepare_body_directory(self, destination: Path) -> Path:
        """Create a private normalized view; never rename licensed originals."""
        import os
        import shutil
        self.validate()
        destination = Path(destination)
        for name, original in self.body_files().items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f'Body asset staging target already exists: {target}')
            try:
                os.link(original, target)
            except OSError:
                shutil.copy2(original, target)
        return destination


class SmplerXAdapter:
    """Load and run a fixed upstream adapter inside the worker process."""

    def __init__(self, assets: SmplerXAssets):
        self.assets = assets
        self._backend: Any | None = None

    def load(self) -> None:
        self.assets.validate()
        from external.smplerx_worker.backend import SmplerXBackend
        self._backend = SmplerXBackend(self.assets)

    def infer(self, rgb, frame_id):
        if self._backend is None:
            raise RuntimeError("SMPLer-X adapter is not loaded")
        return self._backend.infer(rgb, frame_id)

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
