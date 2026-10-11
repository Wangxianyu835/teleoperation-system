"""域随机化模块 - 光照、摩擦、纹理、位置随机化"""

import numpy as np
import pybullet as p


class DomainRandomizer:
    """对仿真环境施加随机化，增强sim-to-real泛化性"""

    def __init__(self):
        # 随机化范围
        self.friction_range = (0.1, 0.8)
        self.object_color_jitter = 0.15   # RGB颜色抖动范围
        self.position_noise = 0.02        # 初始位置微扰（米）

    def randomize(self, task_objects: dict, client_id: int):
        """对任务中的所有物体施加随机化"""
        for name, obj_id in task_objects.items():
            if name == 'table':
                continue  # 操作台不动

            # 1. 摩擦系数随机化
            friction = np.random.uniform(*self.friction_range)
            p.changeDynamics(obj_id, -1, lateralFriction=friction,
                             physicsClientId=client_id)

            # 2. 初始位置微扰
            pos, orn = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=client_id)
            noise = np.random.uniform(-self.position_noise,
                                      self.position_noise, 3)
            noise[2] *= 0.5  # Z轴扰动减小
            p.resetBasePositionAndOrientation(
                obj_id, pos + noise, orn, physicsClientId=client_id)

            # 3. 物体颜色抖动
            self._jitter_color(obj_id, client_id)

    def _jitter_color(self, obj_id: int, client_id: int):
        """对物体的视觉颜色添加随机抖动"""
        try:
            vis_data = p.getVisualShapeData(obj_id,
                                            physicsClientId=client_id)
            if len(vis_data) > 0:
                rgba = list(vis_data[0][7])  # rgbaColor
                if len(rgba) == 4:
                    jittered = [
                        min(1.0, max(0.0, c + np.random.uniform(
                            -self.object_color_jitter, self.object_color_jitter)))
                        for c in rgba[:3]
                    ]
                    jittered.append(rgba[3])
                    p.changeVisualShape(obj_id, -1,
                                        rgbaColor=jittered,
                                        physicsClientId=client_id)
        except Exception:
            pass  # 颜色抖动非必需，失败也不影响

    def get_randomized_params(self) -> dict:
        """返回当前随机化的参数值（用于记录和复现）"""
        return {
            'friction': np.random.uniform(*self.friction_range),
            'position_noise': np.random.uniform(0, self.position_noise),
        }
