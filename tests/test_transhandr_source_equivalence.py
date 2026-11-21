"""Compare the port with numerical statements extracted from local upstream."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline
from teleoperation.contracts.observations import RawHandFrame


UPSTREAM = Path('D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py')


@unittest.skipUnless(UPSTREAM.is_file(), 'local TransHandR reference is unavailable')
class TransHandRSourceEquivalenceTests(unittest.TestCase):
    def test_actual_upstream_extraction_stack_transpose_and_scaling_match_exactly(self):
        tree = ast.parse(UPSTREAM.read_text(encoding='utf-8-sig'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'vision_pro_data_process')
        body = next(n for n in worker.body if isinstance(n, ast.While)).body
        first = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'coordinates' for t in n.targets))
        last = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'relative_coordinates' for t in n.targets))
        function = ast.parse('def points(right_fingers):\n    pass').body[0]
        function.body = deepcopy(body[first:last + 1]) + [ast.Return(ast.Name(id='relative_coordinates', ctx=ast.Load()))]
        namespace = {'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(UPSTREAM), 'exec'), namespace)
        stack = next(n for n in ast.walk(worker) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'np.stack')
        inference = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'redirection_process')
        reshape = [n for n in ast.walk(inference) if
                   (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'reshaped_data' for t in n.targets))
                   or (isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == 'reshaped_data')]
        reshape.sort(key=lambda n: n.lineno)
        rng = np.random.default_rng(1234)
        for scale in (1., 1.7):
            pipeline = TransHandRVisionProPipeline(scale)
            frames = []
            for index in range(12):
                matrices = np.tile(np.eye(4), (25, 1, 1))
                matrices[:, :3, 3] = rng.normal(size=(25, 3))
                frames.append(namespace['points'](matrices))
                frames = frames[-3:]
                raw = RawHandFrame({'left': matrices, 'right': None}, index / 30, 'visionpro',
                                   geometry_encoding={'left': 'transforms25'})
                window = pipeline.process_window(raw)
                if index < 2:
                    self.assertIsNone(window)
                    continue
                environment = {'np': np, 'recent_frames': frames, 'shared_dict': {}, 'scaling_factor': scale}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(stack)], type_ignores=[])), str(UPSTREAM), 'exec'), environment)
                environment['vision_data'] = environment['shared_dict']['vision_pro_data']
                exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(reshape), type_ignores=[])), str(UPSTREAM), 'exec'), environment)
                np.testing.assert_array_equal(window.hands['left'], environment['reshaped_data'][0].astype(np.float32))

    def test_actual_upstream_model_has_identical_outputs_for_the_same_weights(self):
        import importlib.util
        import torch
        from teleoperation.retargeting.hand.transformer import PoseTransformer
        from tests.test_world_hand_pipeline import small_model_kwargs
        source = UPSTREAM.parent / 'model/model_poseformer.py'
        spec = importlib.util.spec_from_file_location('reference_poseformer', source)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = PoseTransformer(**small_model_kwargs()).eval()
        upstream = reference.PoseTransformer(**small_model_kwargs()).eval()
        upstream.load_state_dict(model.state_dict(), strict=True)
        inputs = torch.randn(4, 3, 25, 3)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), upstream(inputs), atol=0, rtol=0)
