import h5py
import torch
import numpy as np
from model.angle2real import create_hand_kinematics
from config.variables_define import *
import torch.nn.functional as F
def angles_to_positions(input_h5_path, output_h5_path, urdf_file=urdf_file, hand_cfg=hand_cfg, 
                       scaling_factor=scaling_factor_rb, axis_correction_matrix=correction_matrix,
                       batch_size=64):
    """
    将关节角度转换为正向运动学的关节位置坐标
    
    Args:
        input_h5_path: 输入的关节角度H5文件路径，格式为(frames, 关节数)
        output_h5_path: 输出的关节位置H5文件路径
        urdf_file: URDF机器人模型文件路径
        hand_cfg: 手部配置文件
        scaling_factor: 缩放因子
        axis_correction_matrix: 轴校正矩阵
        batch_size: 批处理大小，用于提高处理效率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建手部正向运动学模型
    hand_fk = create_hand_kinematics(
        urdf_file, 
        hand_cfg, 
        device, 
        scale_factor=1.0,
        axis_correction_matrix=axis_correction_matrix
    )
    
    # 读取输入的关节角度数据
    with h5py.File(input_h5_path, 'r') as input_file:
        # 假设关节角度存储在'outputs'或类似名称的数据集中
        if 'outputs' in input_file:
            joint_angles = input_file['outputs'][:]
        elif 'angles' in input_file:
            joint_angles = input_file['angles'][:]
        elif 'joint_angles' in input_file:
            joint_angles = input_file['joint_angles'][:]
        else:
            # 如果没有找到标准名称，则尝试获取第一个数据集
            key = list(input_file.keys())[0]
            joint_angles = input_file[key][:]
        
        # 在读取数据后，维度检查之前添加这行
        joint_angles = np.pad(joint_angles, ((0, 0), (0, 5)), mode='constant', constant_values=0)
        print(f"Input shape: {joint_angles.shape}")
        
        # 检查输入数据维度
        if len(joint_angles.shape) != 2:
            raise ValueError(f"Expected input shape (frames, angles), got {joint_angles.shape}")

        num_frames, num_angles = joint_angles.shape
        print(f"Number of frames: {num_frames}, Number of angles: {num_angles}")
    
    # 验证关节角度数量是否正确
    expected_num_angles = len(angle_limit_rob)
    if num_angles != expected_num_angles:
        print(f"Warning: Expected {expected_num_angles} angles, got {num_angles}")
        print("Make sure the input data matches the expected joint configuration.")
    
    # 转换为Tensor并移到设备上
    joint_angles_tensor = torch.from_numpy(joint_angles.astype(np.float32)).to(device)
    # 添加批次维度 (frames, angles) -> (frames, 1, angles)
    joint_angles_tensor = joint_angles_tensor.unsqueeze(1)
    
    # 存储所有位置结果
    all_positions = []
    
    # 分批处理数据以节省内存
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx + batch_size, num_frames)
        batch_angles = joint_angles_tensor[start_idx:end_idx]
        
        with torch.no_grad():
            # 计算正向运动学得到关节位置
            _,_,batch_positions = hand_fk.forward(batch_angles)
            
            # 将结果移到CPU并转换为numpy
            batch_positions_np = batch_positions.cpu().numpy()
            
            # 移除批次维度并添加到结果列表
            # batch_positions_np = batch_positions_np.squeeze(1)  # 移除第1维（批次维）
            all_positions.append(batch_positions_np)
        
        # 显示进度
        if start_idx % (batch_size * 10) == 0:
            print(f"Processed {end_idx}/{num_frames} frames")
    
    print(f"Processing completed. Total frames: {num_frames}")
    
    # 将所有位置数据合并成一个数组
    if all_positions:
        positions_array = np.concatenate(all_positions, axis=0)
        print(f"Output shape: {positions_array.shape}")
        
        # 保存到输出的H5文件
        with h5py.File(output_h5_path, 'w') as output_file:
            output_file.create_dataset('joint_positions', data=positions_array)
            output_file.attrs['num_frames'] = positions_array.shape[0]
            output_file.attrs['num_joints'] = positions_array.shape[1] if len(positions_array.shape) > 1 else 0
            output_file.attrs['coordinate_dims'] = positions_array.shape[2] if len(positions_array.shape) > 2 else 0
            output_file.attrs['description'] = 'Joint positions calculated from forward kinematics based on input joint angles'
            output_file.attrs['input_file'] = input_h5_path
        
        print(f"Joint positions saved to {output_h5_path}")
    else:
        print("No positions were calculated")


def convert_single_frame_angles_to_positions(joint_angles, urdf_file=urdf_file, hand_cfg=hand_cfg,
                                          scaling_factor=scaling_factor_rb, axis_correction_matrix=correction_matrix):
    """
    转换单帧关节角度到关节位置
    
    Args:
        joint_angles: 单帧关节角度，形状为 (num_angles,) 或 (1, num_angles)
        其他参数同上
    
    Returns:
        对应的关节位置，形状为 (num_joints, 3)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建手部正向运动学模型
    hand_fk = create_hand_kinematics(
        urdf_file, 
        hand_cfg, 
        device, 
        scale_factor=scaling_factor,
        axis_correction_matrix=axis_correction_matrix
    )
    
    # 确保输入是正确的形状
    if isinstance(joint_angles, np.ndarray):
        joint_angles = torch.from_numpy(joint_angles.astype(np.float32))
    
    if len(joint_angles.shape) == 1:
        joint_angles = joint_angles.unsqueeze(0).unsqueeze(0)  # (num_angles,) -> (1, 1, num_angles)
    elif len(joint_angles.shape) == 2:
        if joint_angles.shape[0] == 1:
            joint_angles = joint_angles.unsqueeze(0)  # (1, num_angles) -> (1, 1, num_angles)
        else:
            joint_angles = joint_angles.unsqueeze(1)  # (frames, num_angles) -> (frames, 1, num_angles)
    
    joint_angles = joint_angles.to(device)
    
    with torch.no_grad():
        joint_positions = hand_fk.forward_kinematics(joint_angles)
        positions_np = joint_positions.cpu().numpy()
    
    # 移除不必要的维度
    if len(positions_np.shape) == 3 and positions_np.shape[0] == 1:
        positions_np = positions_np.squeeze(0)
    
    return positions_np


# def main():
#     """
#     主函数，用于执行从关节角度到位置的转换
#     """
#     # 设置输入输出路径
#     input_h5_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output.h5"
#     output_h5_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_positions_output.h5"
    
#     # 执行转换
#     angles_to_positions(
#         input_h5_path=input_h5_path,
#         output_h5_path=output_h5_path,
#         batch_size=64  # 可根据内存情况调整批次大小
#     )


if __name__ == "__main__":
    # 使用示例
    # 从您之前运行的模型输出文件作为输入
    # 消融实验类型
    # ab_experiment_name = 'None'
    # ab_experiment_name = 'ab_vec_loss'
    ab_experiment_name = 'ab_tip_pos_loss'
    # ab_experiment_name = 'ab_col_loss'
    # ab_experiment_name = 'ab_tip_dis_loss'
    # # 'yumi'  'linker'  'shadow' 'svhhand' 'inspire'
    # hand_brand = 'shadow'  # 替换为实际的手部品牌
    # input_h5 = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output.h5"  # 修改为您的实际输入文件
    # output_h5 = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_positions_output.h5"  # 修改为您的实际输出文件
    # input_h5 = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output2.h5"  # 修改为您的实际输入文件
    # output_h5 = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_positions_slahmr_output2.h5"  # 修改为您的实际输出文件
    input_h5 = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_output.h5"  # 修改为您的实际输入文件
    output_h5 = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_positions_output.h5"  # 修改为您的实际输出文件

    
    angles_to_positions(
        input_h5_path=input_h5,
        output_h5_path=output_h5,
        batch_size=64  # 根据您的硬件资源调整批次大小
    )
    
    print("Conversion completed!")