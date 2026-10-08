"""Existing native action file schemas and serialization."""
import os
import numpy as np

def load_action_file(path):
    """读取契约H 的动作序列文件，返回 (actions, meta)"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到动作文件：{path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == '.npz':
        z = np.load(path, allow_pickle=True)
        if 'actions' not in z.files:
            raise ValueError("npz 中缺少必需的 'actions' 数组")
        actions = np.asarray(z['actions'], dtype=np.float32)
        meta = {}
        for k in z.files:
            if k == 'actions':
                continue
            v = z[k]
            meta[k] = v.item() if getattr(v, 'shape', ()) == () else v
        return actions, meta

    if ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            if 'actions' not in f:
                raise ValueError("h5 中缺少必需的 'actions' 数据集")
            actions = np.asarray(f['actions'][:], dtype=np.float32)
            meta = {}
            for k, v in f.attrs.items():
                meta[k] = v.decode() if isinstance(v, bytes) else v
            if 'timestamps' in f:
                meta['timestamps'] = np.asarray(f['timestamps'][:])
        return actions, meta

    raise ValueError(f"不支持的文件格式：{ext}（请用 .npz / .h5 / .hdf5）")


def save_action_file(path, actions, meta):
    """按契约H 写出动作序列文件（用于生成示例数据）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    meta = dict(meta)

    if ext == '.npz':
        payload = {'actions': np.asarray(actions, dtype=np.float32)}
        payload.update(meta)
        np.savez_compressed(path, **payload)
    elif ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'w') as f:
            f.create_dataset('actions',
                             data=np.asarray(actions, dtype=np.float32))
            if 'timestamps' in meta:
                f.create_dataset(
                    'timestamps',
                    data=np.asarray(meta.pop('timestamps'), dtype=np.float64))
            for k, v in meta.items():
                f.attrs[k] = v
    else:
        raise ValueError(f"不支持的文件格式：{ext}")

    print(f"  已写出：{path}  ({os.path.getsize(path)/1024:.1f} KB)")

