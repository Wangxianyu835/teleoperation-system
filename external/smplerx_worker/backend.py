"""S32 inference boundary for upstream commit 064baef0.

Keep upstream network, checkpoint key conversion and crop math unchanged.
Application-specific changes: explicit asset roots, no training dataset imports,
one unambiguous person, and derived joints/rotations from the same SMPL-X pass.
"""
from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
import sys
import tempfile
import time


class SmplerXBackend:
    def __init__(self, assets):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('SMPLer-X S32 requires CUDA in the isolated worker environment')
        self.torch = torch
        torch.set_num_threads(2)
        self._body_directory = tempfile.TemporaryDirectory(prefix='smplerx-body-')
        try:
            self._initialize(assets)
        except BaseException:
            self._body_directory.cleanup()
            raise
        self._session = None
        self._shape = None
        self._operator = None

    def _initialize(self, assets):
        torch = self.torch
        root = assets.source_root.resolve()
        for relative in ('common', 'main', 'main/transformer_utils'):
            sys.path.insert(0, str(root / relative))
        from config import cfg
        cfg.get_config_fromfile(str(root / 'main/config/config_smpler_x_s32.py'))
        cfg.human_model_path = str(assets.prepare_body_directory(Path(self._body_directory.name)))
        cfg.encoder_config_file = str(root / 'main' / cfg.encoder_config_file)
        cfg.pretrained_model_path = str(assets.checkpoint.resolve())
        cfg.testset = 'EHF'
        cfg.num_gpus = 1
        self.cfg = cfg
        # Official inference.py imports MMDetection before the custom MMPose
        # registrations. Reversing that order causes duplicate registry keys.
        from mmdet.apis import init_detector, inference_detector
        from external.smplerx_worker.torchgeometry_compat import configure_torchgeometry
        configure_torchgeometry()
        # Importing common.base would import training datasets and evaluation
        # assets. Reproduce only Demoer._make_model's graph and weight loading.
        from SMPLer_X import get_model
        self.model = torch.nn.DataParallel(get_model('test')).cuda()
        checkpoint = torch.load(str(assets.checkpoint), map_location='cpu')
        converted = {}
        for key, value in checkpoint['network'].items():
            if not key.startswith('module.'):
                key = 'module.' + key
            key = key.replace('module.backbone', 'module.encoder').replace(
                'body_rotation_net', 'body_regressor').replace('hand_rotation_net', 'hand_regressor')
            converted[key] = value
        incompatible = self.model.load_state_dict(converted, strict=False)
        parameters = set(dict(self.model.named_parameters()))
        missing_parameters = parameters.intersection(incompatible.missing_keys)
        if missing_parameters:
            raise RuntimeError(f'S32 checkpoint misses learned parameters: {sorted(missing_parameters)}')
        self.model.eval()
        self.checkpoint_missing_buffers = list(incompatible.missing_keys)
        del checkpoint, converted
        with assets.checkpoint.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        self.model_hash = digest.hexdigest()
        detector_root = root / 'pretrained_models/mmdet'
        detector_config = detector_root / 'mmdet_faster_rcnn_r50_fpn_coco.py'
        detector_checkpoint = detector_root / 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        for path in (detector_config, detector_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'SMPLer-X detector asset not found: {path}')
        self.detector = init_detector(str(detector_config), str(detector_checkpoint), device='cuda:0')
        self.detect = inference_detector
        from utils.preprocessing import process_bbox, generate_patch_image
        self.process_bbox, self.generate_patch = process_bbox, generate_patch_image

    def _invalid(self, frame_id, reason):
        from teleoperation.contracts.body import SmplerXResult
        return SmplerXResult(frame_id, time.monotonic(), 's32', self.model_hash, {},
                             {'left': False, 'right': False}, reason=reason)

    def infer(self, rgb, frame_id):
        import numpy as np
        from teleoperation.contracts.body import SmplerXResult
        from external.smplerx_worker.geometry import derive_body_arrays
        if self._session != frame_id.session_id:
            from external.smplerx_worker.operator_state import OperatorState
            self._session = frame_id.session_id
            self._shape = None
            self._operator = OperatorState()
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('worker input must be uint8 RGB')
        age = time.monotonic() - frame_id.capture_time_s
        if age < 0 or age > .5:
            return self._invalid(frame_id, 'input frame expired before inference')
        # MMDetection receives BGR; upstream load_img/crop and the network use RGB.
        detections = np.asarray(self.detect(self.detector, rgb[..., ::-1].copy())[0])
        if detections.size == 0:
            return self._invalid(frame_id, self._operator.person(None, frame_id.capture_time_s))
        detections = detections[(detections[:, 4] >= .5)
                                & (detections[:, 2] - detections[:, 0] >= 50)
                                & (detections[:, 3] - detections[:, 1] >= 150)]
        # Never choose a different operator merely because its box is larger.
        if len(detections) != 1:
            return self._invalid(frame_id, self._operator.person(None, frame_id.capture_time_s,
                                                                ambiguous=len(detections) > 1))
        reason = self._operator.person(detections[0, :4], frame_id.capture_time_s)
        if reason:
            return self._invalid(frame_id, reason)
        box = detections[0, :4].copy()
        box[2:] -= box[:2]
        height, width = rgb.shape[:2]
        bbox = self.process_bbox(box, width, height)
        if bbox is None:
            return self._invalid(frame_id, 'invalid person crop')
        patch, _, inverse = self.generate_patch(rgb.astype(np.float32), bbox, 1., 0., False,
                                                self.cfg.input_img_shape)
        torch = self.torch
        image = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float().div(255).cuda()[None]
        with torch.inference_mode():
            out = self.model({'img': image}, {}, {}, 'test')
            if self._shape is None:
                shape = self._operator.shape_sample(out['smplx_shape'].detach().cpu().numpy()[0])
                if shape is None:
                    return self._invalid(frame_id, 'calibrating body shape: hold a neutral pose')
                self._shape = torch.as_tensor(shape, device=image.device).reshape(1, 10)
            layer = self.model.module.smplx_layer
            zeros = torch.zeros((1, 3), device=image.device)
            derived = layer(
                betas=self._shape, global_orient=out['smplx_root_pose'],
                body_pose=out['smplx_body_pose'], left_hand_pose=out['smplx_lhand_pose'],
                right_hand_pose=out['smplx_rhand_pose'], jaw_pose=out['smplx_jaw_pose'],
                leye_pose=zeros, reye_pose=zeros, expression=out['smplx_expr'],
                return_full_pose=True)
        cpu = lambda tensor: tensor.detach().cpu().numpy()[0]
        parents = layer.parents.detach().cpu().numpy()
        arrays = derive_body_arrays(
            cpu(derived.joints)[:len(parents)], cpu(derived.full_pose), parents,
            cpu(out['cam_trans']), self.cfg.focal, self.cfg.princpt,
            self.cfg.input_body_shape, self.cfg.input_img_shape, inverse, rgb.shape[:2])
        for key in ('smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose',
                    'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'cam_trans'):
            arrays[key] = cpu(out[key])
        arrays['fixed_shape'] = cpu(self._shape)
        arrays['bbox_xywh'] = np.asarray(bbox)
        arrays['inverse_crop'] = inverse
        return SmplerXResult(frame_id, time.monotonic(), 's32', self.model_hash, arrays,
                             {'left': True, 'right': True}, metadata={
                                 'joint_indices': {'left': {'shoulder': 16, 'elbow': 18, 'wrist': 20},
                                                   'right': {'shoulder': 17, 'elbow': 19, 'wrist': 21}},
                                 'upstream_commit': '064baef0e4ab5277a3297691bc1d46ea5412586f',
                                 'shape_policy': 'median_of_five_stable_samples_then_frozen',
                                 'checkpoint_missing_buffers': self.checkpoint_missing_buffers})

    def close(self):
        self.detector = self.model = None
        self._body_directory.cleanup()
