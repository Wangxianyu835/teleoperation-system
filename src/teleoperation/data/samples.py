import os
import numpy as np

def write_hand_file(path, kp3d, kp2d, wrist, timestamps,
                    side='right', fps=30.0, source='synthetic'):
    """按契约G 写出人类手部数据文件"""
    import h5py
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('keypoints_3d', data=kp3d)
        f.create_dataset('keypoints_2d', data=kp2d)
        f.create_dataset('wrist_pose', data=wrist)
        f.create_dataset('timestamps', data=timestamps)
        f.attrs['hand_side'] = side
        f.attrs['fps'] = float(fps)
        f.attrs['source'] = source
        f.attrs['mediapipe_version'] = 'synthetic-1.0'
        f.attrs['landmark_order'] = ','.join(MEDIAPIPE_HAND_LANDMARKS)
    print(f"  已写出：{path}  ({os.path.getsize(path)/1024:.1f} KB)")


MEDIAPIPE_HAND_LANDMARKS = 21
