"""
Forward Kinematics Module for converting joint angles to 3D positions and orientations
"""
import torch
from urchin import URDF, matrix_to_xyz_rpy
import numpy as np

class ForwardKinematics:
    """
    正向运动学模块，将关节角度转换为3D位置和方向
    """
    
    def __init__(self, joint_names, joint_parents, joint_offsets, joint_axes, device=None, scaling_factor=1.0,axis_correction_matrix=None):
        """
        初始化正向运动学模块
        
        参数:
            joint_names: 关节名称列表
            joint_parents: 关节父节点列表，joint_parents[i]表示关节i的父节点索引(-1表示根节点)
            joint_offsets: 关节偏移量列表 [xyzrpy]，每个关节的xyz位置和rpy旋转
            joint_axes: 关节旋转轴列表
            device: 设备类型 (如 'cuda' 或 'cpu')
        """
        self.joint_names = joint_names
        self.joint_parents = joint_parents
        # 从numpy数组列表创建tensor非常缓慢，建议先使用numpy.array()转换为单个numpy数组，然后再转为tensor。
        joint_offsets_array = np.array(joint_offsets)
        self.joint_offsets = torch.tensor(joint_offsets_array, dtype=torch.float32)
        joint_axes_array = np.array(joint_axes)
        self.joint_axes = torch.tensor(joint_axes_array, dtype=torch.float32)
        self.num_joints = len(joint_names)
        self.scaling_factor = scaling_factor
        if axis_correction_matrix is not None:
            self.axis_correction_matrix = axis_correction_matrix
            self.axis_correction_matrix = self.axis_correction_matrix.to(torch.get_default_device() if hasattr(torch, 'get_default_device') else 'cpu')
        else:
            self.axis_correction_matrix = None
        # 将参数移动到指定设备
        if device is not None:
            self.joint_offsets = self.joint_offsets.to(device)
            self.joint_axes = self.joint_axes.to(device)
        else:
            # 默认情况下，使用当前默认的设备
            self.joint_offsets = self.joint_offsets.to(torch.get_default_device() if hasattr(torch, 'get_default_device') else 'cpu')
            self.joint_axes = self.joint_axes.to(torch.get_default_device() if hasattr(torch, 'get_default_device') else 'cpu')

    def forward(self, joint_angles):
        """
        正向运动学计算
        
        参数:
            joint_angles: 形状为 [batch_size, num_joints] 的张量，包含关节角度（弧度）
            
        返回:
            positions: 形状为 [batch_size, num_joints, 3] 的张量，包含3D位置（相对于局部坐标系）
            orientations: 形状为 [batch_size, num_joints, 3, 3] 的张量，包含旋转矩阵
            global_positions: 形状为 [batch_size, num_joints, 3] 的张量，包含全局位置
        """
        batch_size = joint_angles.shape[0]
        
        # 确保joint_angles具有正确的形状
        if len(joint_angles.shape) == 2:
            # 如果输入是 [batch_size, num_joints]，确保它是正确的
            pass
        elif len(joint_angles.shape) == 3:
            # 如果输入是 [batch_size, 1, num_joints]，需要调整
            if joint_angles.shape[1] == 1:
                joint_angles = joint_angles.squeeze(1)  # [batch_size, num_joints]
            elif joint_angles.shape[2] == 1:
                joint_angles = joint_angles.squeeze(2)  # [batch_size, num_joints]
        
        # 确保参数与输入张量在同一设备上
        device = joint_angles.device
        joint_offsets = self.joint_offsets.to(device)
        joint_axes = self.joint_axes.to(device)
        
        # 初始化张量
        positions = torch.zeros(batch_size, self.num_joints, 3, device=device)
        global_positions = torch.zeros(batch_size, self.num_joints, 3, device=device)
        rot_matrices = torch.zeros(batch_size, self.num_joints, 3, 3, device=device)
        
        # 从偏移量中提取xyz和rpy
        xyz = joint_offsets[:, :3]  # [num_joints, 3]
        rpy = joint_offsets[:, 3:]  # [num_joints, 3]
        
        # 计算初始关节方向的旋转矩阵
        rpy_transforms = self._transform_from_euler_batch(rpy.unsqueeze(0).expand(batch_size, -1, -1))
        
        # 计算关节角度的旋转矩阵
        # 标准化轴并乘以角度
        axis_norms = torch.norm(joint_axes, dim=-1, keepdim=True)
        axis_norms = axis_norms.unsqueeze(0).expand(batch_size, -1, -1)  # [1, num_joints, 1]
        joint_angles_expanded = joint_angles.unsqueeze(-1)  # [batch_size, num_joints, 1]
        normalized_angles = joint_angles_expanded * axis_norms  # [batch_size, num_joints, 1]
        angle_transforms = self._transform_from_multiple_axis(normalized_angles, 
                                                              joint_axes.unsqueeze(0).expand(batch_size, -1, -1))
        
        # 按拓扑顺序处理每个关节
        for joint_idx in range(self.num_joints):
            parent_idx = self.joint_parents[joint_idx]
            
            if parent_idx != -1:
                # 计算相对于父节点的局部位置
                local_xyz = xyz[joint_idx].unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 1, 3]
                parent_rot = rot_matrices[:, parent_idx, :, :].clone()  # [batch_size, 3, 3]
                
                # 通过父节点的旋转变换局部xyz
                rotated_xyz = torch.bmm(parent_rot, local_xyz.transpose(1, 2)).transpose(1, 2).squeeze(1)  # [batch_size, 3]
                
                # 计算相对于根节点的位置
                positions[:, joint_idx, :] = rotated_xyz + positions[:, parent_idx, :]
                # 计算全局位置
                global_positions[:, joint_idx, :] = rotated_xyz + global_positions[:, parent_idx, :]
                
                # 计算旋转矩阵
                rpy_transform = rpy_transforms[:, joint_idx, :, :]  # [batch_size, 3, 3]
                angle_transform = angle_transforms[:, joint_idx, :, :]  # [batch_size, 3, 3]
                
                # 结合父节点旋转和关节旋转
                rot_matrices[:, joint_idx, :, :] = torch.bmm(parent_rot, 
                                                            torch.bmm(rpy_transform, angle_transform))
            else:
                # 根关节
                positions[:, joint_idx, :] = torch.zeros(3, device=device)
                global_positions[:, joint_idx, :] = xyz[joint_idx].unsqueeze(0).expand(batch_size, -1)
                
                # 根节点旋转矩阵
                rpy_transform = rpy_transforms[:, joint_idx, :, :]  # [batch_size, 3, 3]
                angle_transform = angle_transforms[:, joint_idx, :, :]  # [batch_size, 3, 3]
                rot_matrices[:, joint_idx, :, :] = torch.bmm(rpy_transform, angle_transform)
        
        # 应用缩放因子到位置
        positions = positions * self.scaling_factor
        global_positions = global_positions * self.scaling_factor
        # 将结果乘上一个矩阵以调整坐标系（如果需要）
        # 例如将x，y，z转为-y，-x ，z 需要乘以以下矩阵
        # correction = torch.tensor([[0, -1, 0],
        #                            [-1, 0, 0],
        #                            [0, 0, 1]], dtype=torch.float32)
        # 将结果乘上一个矩阵以调整坐标系（如果需要）
        batch_size, num_joints, _ = positions.shape
        # 注意：这里需要正确处理张量维度
        if self.axis_correction_matrix is not None:
            correction = self.axis_correction_matrix.to(device)
            # 对旋转矩阵应用校正
            rot_matrices = torch.matmul(correction.unsqueeze(0).unsqueeze(0), rot_matrices)
            # 对位置应用校正：修正矩阵乘法的维度
            # positions: [batch_size, num_joints, 3] -> [batch_size*num_joints, 3]
            positions_flat = positions.view(-1, 3)  # [batch_size*num_joints, 3]
            corrected_positions_flat = torch.matmul(positions_flat, correction.T)  # [batch_size*num_joints, 3]
            positions = corrected_positions_flat.view(batch_size, num_joints, 3)  # [batch_size, num_joints, 3]
            
            # 对全局位置应用同样的校正
            global_positions_flat = global_positions.view(-1, 3)  # [batch_size*num_joints, 3]
            corrected_global_positions_flat = torch.matmul(global_positions_flat, correction.T)  # [batch_size*num_joints, 3]
            global_positions = corrected_global_positions_flat.view(batch_size, num_joints, 3)  # [batch_size, num_joints, 3]

        return positions, rot_matrices, global_positions

    
    @staticmethod
    def _transform_from_euler_batch(angles, order='xyz'):
        """
        将欧拉角转换为批量旋转矩阵
        
        参数:
            angles: 形状为 [batch_size, num_joints, 3] 的张量，包含 [roll, pitch, yaw]
            order: 旋转顺序（默认 'xyz'）
            
        返回:
            形状为 [batch_size, num_joints, 3, 3] 的旋转矩阵
        """
        # 提取各个角度
        roll = angles[:, :, 0]  # [batch_size, num_joints]
        pitch = angles[:, :, 1]
        yaw = angles[:, :, 2]
        
        # 为每个角度计算sin和cos
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        
        # 为ZYX顺序构建旋转矩阵（相当于逆序的XYZ）
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R = torch.zeros(angles.shape[0], angles.shape[1], 3, 3, device=angles.device)
        
        R[:, :, 0, 0] = cy * cp
        R[:, :, 0, 1] = cy * sp * sr - sy * cr
        R[:, :, 0, 2] = cy * sp * cr + sy * sr
        R[:, :, 1, 0] = sy * cp
        R[:, :, 1, 1] = sy * sp * sr + cy * cr
        R[:, :, 1, 2] = sy * sp * cr - cy * sr
        R[:, :, 2, 0] = -sp
        R[:, :, 2, 1] = cp * sr
        R[:, :, 2, 2] = cp * cr
        
        return R
    
    @staticmethod
    def _transform_from_multiple_axis(angles, axes):
        """
        从轴角表示创建旋转矩阵
        
        参数:
            angles: 形状为 [batch_size, num_joints, 1] 的张量，包含角度
            axes: 形状为 [batch_size, num_joints, 3] 的张量，包含旋转轴
            
        返回:
            形状为 [batch_size, num_joints, 3, 3] 的旋转矩阵
        """
        # 标准化轴
        axes_norm = torch.norm(axes, dim=-1, keepdim=True)
        normalized_axes = axes / (axes_norm + 1e-8)  # 添加小值以避免除零错误
        
        # 提取分量
        n1 = normalized_axes[..., 0]  # [batch_size, num_joints]
        n2 = normalized_axes[..., 1]
        n3 = normalized_axes[..., 2]
        angle = angles.squeeze(-1)  # [batch_size, num_joints]
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one_minus_cos = 1 - cos
        
        # 使用Rodrigues公式构建旋转矩阵
        R = torch.zeros(angles.shape[0], angles.shape[1], 3, 3, device=angles.device)
        
        R[..., 0, 0] = cos + n1 * n1 * one_minus_cos
        R[..., 0, 1] = n1 * n2 * one_minus_cos - n3 * sin
        R[..., 0, 2] = n1 * n3 * one_minus_cos + n2 * sin
        R[..., 1, 0] = n1 * n2 * one_minus_cos + n3 * sin
        R[..., 1, 1] = cos + n2 * n2 * one_minus_cos
        R[..., 1, 2] = n2 * n3 * one_minus_cos - n1 * sin
        R[..., 2, 0] = n1 * n3 * one_minus_cos - n2 * sin
        R[..., 2, 1] = n2 * n3 * one_minus_cos + n1 * sin
        R[..., 2, 2] = cos + n3 * n3 * one_minus_cos
        
        return R


def _joint_record(joint):
    axis = joint.axis if joint.axis is not None else np.zeros(3, dtype=np.float64)
    return {
        'type': joint.joint_type,
        'axis': axis,
        'parent': joint.parent,
        'child': joint.child,
        'origin': matrix_to_xyz_rpy(joint.origin),
        'lower': joint.limit.lower if joint.limit else 0,
        'upper': joint.limit.upper if joint.limit else 0,
    }


def parse_urdf_to_joints(urdf_file, cfg):
    """
    从URDF文件和配置解析关节信息
    
    参数:
        urdf_file: URDF文件路径
        cfg: 配置字典，包含关节名称、边等信息
    返回:
        joint_names: 关节名称列表
        joint_parents: 关节父节点列表
        joint_offsets: 关节偏移量列表 [xyzrpy]
        joint_axes: 关节旋转轴列表
    """
    # 加载URDF
    # FK only needs joint metadata; mesh loading would make relative mesh
    # paths depend on the process working directory.
    robot = URDF.load(urdf_file, lazy_load_meshes=True)

    # 解析关节参数
    joints = {joint.name: _joint_record(joint) for joint in robot.joints}
    links = {link.name for link in robot.links}

    # 收集关节名称和索引映射
    joints_name = cfg['joints_name']
    joints_index = {name: i for i, name in enumerate(joints_name)}

    # 构建父子关系
    joint_parents = [-1] * len(joints_name)  # 初始化为-1（表示根节点）
    for edge in cfg['edges']:
        parent_name, child_name = edge
        parent_idx = joints_index[parent_name]
        child_idx = joints_index[child_name]
        joint_parents[child_idx] = parent_idx

    # A configured root can name the URDF's root link instead of a joint. In
    # that case it is a zero-transform logical node in the FK graph.
    root_name = cfg.get('root_name')
    resolved_joints = {}
    missing_joints = []
    for joint_name in joints_name:
        if joint_name in joints:
            resolved_joints[joint_name] = joints[joint_name]
        elif joint_name == root_name and joint_name in links:
            resolved_joints[joint_name] = {
                'type': 'fixed',
                'axis': np.zeros(3, dtype=np.float64),
                'parent': None,
                'child': joint_name,
                'origin': np.zeros(6, dtype=np.float64),
                'lower': 0,
                'upper': 0,
            }
        else:
            missing_joints.append(joint_name)

    if missing_joints:
        missing = ', '.join(missing_joints)
        raise ValueError(
            f"URDF {urdf_file} is missing configured joints: {missing}. "
            "Only the configured root link can be supplied implicitly."
        )

    # 获取关节偏移量和轴
    joint_offsets = []
    joint_axes = []
    for joint_name in joints_name:
        joint = resolved_joints[joint_name]
        joint_offsets.append(joint['origin'])
        # 根节点和固定关节不接受模型输出角度。
        if joint_name == root_name or joint['type'] == 'fixed':
            joint_axes.append([0.0, 0.0, 0.0])
        else:
            joint_axes.append(joint['axis'])

    return joints_name, joint_parents, joint_offsets, joint_axes


def create_hand_kinematics(
    urdf_file,
    cfg,
    device=None,
    scale_factor=1.0,
    axis_correction_matrix=None,
):
    """
    创建手部运动学模块
    
    参数:
        urdf_file: URDF文件路径
        cfg: 配置字典
        device: 设备类型 (如 'cuda' 或 'cpu')
    返回:
        ForwardKinematics对象
    """
    print("URDF:", urdf_file)
    joint_names, joint_parents, joint_offsets, joint_axes = parse_urdf_to_joints(
        urdf_file,
        cfg,
    )
    
    # 创建并返回ForwardKinematics对象
    return ForwardKinematics(joint_names, joint_parents, joint_offsets, joint_axes, device, scaling_factor=scale_factor, axis_correction_matrix=axis_correction_matrix)

