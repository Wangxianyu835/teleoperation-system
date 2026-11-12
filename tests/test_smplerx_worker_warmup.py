"""Startup warmup completes before operator identity is established."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

class WorkerWarmupTests(unittest.TestCase):
    def test_detector_network_and_derived_layer_warm_before_any_operator_sample(self):
        import torch
        from external.smplerx_worker.backend import SmplerXBackend
        calls=[]
        class Network(torch.nn.Module):
            def forward(self,inputs,targets,metadata,mode):
                calls.append('network')
                self.assert_input=inputs['img'].clone()
                return {name:torch.zeros((1,size)) for name,size in {
                    'smplx_shape':10,'smplx_root_pose':3,'smplx_body_pose':63,
                    'smplx_lhand_pose':45,'smplx_rhand_pose':45,'smplx_jaw_pose':3,'smplx_expr':10}.items()}
        class Layer:
            def __call__(self,**kwargs):
                calls.append('derived')
                self.kwargs=kwargs
                return object()
        layer=Layer();network=Network()
        network.module=SimpleNamespace(smplx_layer=layer)
        backend=SmplerXBackend.__new__(SmplerXBackend)
        backend.torch=torch;backend.cfg=SimpleNamespace(input_img_shape=(32,24))
        backend.model=network;backend.detector=object()
        backend.detect=lambda detector,image:calls.append(('detector',image.shape,image.dtype))
        zeros=torch.zeros
        def cpu_zeros(*args,**kwargs):
            kwargs['device']='cpu'
            return zeros(*args,**kwargs)
        with patch.object(torch,'zeros',side_effect=cpu_zeros),patch.object(torch.cuda,'synchronize',side_effect=lambda:calls.append('synchronized')):
            backend._warmup()
        self.assertEqual(calls,[('detector',(32,24,3),np.dtype('uint8')),'network','derived','synchronized'])
        self.assertEqual(tuple(network.assert_input.shape),(1,3,32,24))
        self.assertFalse(network.assert_input.requires_grad)
        self.assertTrue(layer.kwargs['return_full_pose'])
        self.assertFalse(hasattr(backend,'_operator'))
        self.assertFalse(hasattr(backend,'_shape'))
        self.assertGreaterEqual(backend.startup_warmup_s,0)
