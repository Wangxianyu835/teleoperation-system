"""Validate body asset discovery without importing a model framework."""
import tempfile
import unittest
from pathlib import Path
from external.smplerx_worker.adapter import SmplerXAssets


class SmplerXAssetsTests(unittest.TestCase):
    def test_staging_normalizes_names_and_preserves_original_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 's32.pth.tar'
            checkpoint.touch()
            assets = SmplerXAssets(root, checkpoint, root)
            for path in assets.body_files().values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'body-asset')
            normalized = assets.prepare_body_directory(root / 'staged')
            self.assertEqual((normalized / 'smpl/SMPL_NEUTRAL.pkl').read_bytes(), b'body-asset')
            self.assertTrue((root / 'smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl').is_file())
            self.assertFalse((root / 'smpl/SMPL_NEUTRAL.pkl').exists())

    def test_existing_empty_directories_are_not_valid_model_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 's32.pth.tar'
            checkpoint.touch()
            assets = SmplerXAssets(root, checkpoint, root)
            with self.assertRaisesRegex(FileNotFoundError, 'SMPLX_NEUTRAL.npz'):
                assets.validate()

    def test_legacy_smpl_names_resolve_without_renaming_user_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'smpl').mkdir()
            for short in ('neutral', 'm', 'f'):
                (root / 'smpl' / f'basicmodel_{short}_lbs_10_207_0_v1.1.0.pkl').touch()
            assets = SmplerXAssets(root, root / 'checkpoint', root)
            files = assets.body_files()
            self.assertEqual(files['smpl/SMPL_MALE.pkl'].name, 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl')
            self.assertEqual(files['smpl/SMPL_FEMALE.pkl'].name, 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl')
            self.assertFalse((root / 'smpl' / 'SMPL_MALE.pkl').exists())
