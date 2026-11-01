"""Keep torchgeometry 0.1.2 math while fixing obsolete boolean subtraction.

The installed third-party source remains unchanged. For boolean masks,
``1 - mask`` and ``~mask`` select exactly the same entries; current torch only
accepts the latter. Fail on an unexpected dependency rather than guessing.
"""
import inspect


def configure_torchgeometry():
    import torchgeometry
    from torchgeometry.core import conversions
    original = conversions.rotation_matrix_to_quaternion
    if getattr(original, '_smplerx_mask_compatible', False):
        return
    if torchgeometry.__version__ != '0.1.2':
        raise RuntimeError('SMPLer-X torchgeometry compatibility requires version 0.1.2')
    source = inspect.getsource(original)
    for name, count in (('mask_d0_d1', 1), ('mask_d2', 2), ('mask_d0_nd1', 1)):
        old = f'(1 - {name})'
        if source.count(old) != count:
            raise RuntimeError('Unexpected torchgeometry mask implementation')
        source = source.replace(old, f'(~{name})')
    namespace = dict(conversions.__dict__)
    exec(compile(source, '<smplerx-torchgeometry-mask-compat>', 'exec'), namespace)
    patched = namespace['rotation_matrix_to_quaternion']
    patched._smplerx_mask_compatible = True
    conversions.rotation_matrix_to_quaternion = patched
    torchgeometry.rotation_matrix_to_quaternion = patched
