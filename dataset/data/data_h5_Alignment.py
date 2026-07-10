import h5py
import numpy as np

def align_coordinate_system_for_subject(file_path, subject_key, output_file_path=None):
    """
    按照 z·=-x, y·=z, x·=-y 的变换规则对齐坐标系
    
    参数:
    file_path: h5文件路径
    subject_key: 主题键值（一级key）
    output_file_path: 输出文件路径（可选）
    
    返回:
    aligned_data: 坐标系对齐后的数据
    """
    # 打开原始h5文件并一次性读取所有需要的数据
    with h5py.File(file_path, 'r') as h5_file:
        # 检查数据是否存在
        if subject_key not in h5_file:
            raise KeyError(f"键 '{subject_key}' 不存在于文件中")
        
        data_group = h5_file[subject_key]
        
        # 检查所需数据集是否存在
        required_datasets = ['r_glove_pos']
        missing_datasets = [ds for ds in required_datasets if ds not in data_group]
        if missing_datasets:
            print(f"警告: 在主题 '{subject_key}' 中缺少数据集: {missing_datasets}")
            return None, None
        
        # 读取r_glove_pos数据
        r_glove_pos_data = data_group['r_glove_pos'][:]
        
        # 检查是否有l_glove_pos数据
        has_left_hand = 'l_glove_pos' in data_group
        if has_left_hand:
            l_glove_pos_data = data_group['l_glove_pos'][:]
        else:
            # 如果没有左手数据，创建零矩阵
            l_glove_pos_data = np.zeros_like(r_glove_pos_data)
        
        # 检查是否有frame_ids
        has_frame_ids = 'frame_ids' in data_group
        if has_frame_ids:
            frame_ids = data_group['frame_ids'][:]
        else:
            # 如果没有frame_ids，生成默认的frame_ids
            frame_ids = np.arange(r_glove_pos_data.shape[0]).reshape(-1, 1)
        
        # 检查数据维度是否适合进行坐标变换
        if r_glove_pos_data.ndim < 2 or r_glove_pos_data.shape[-1] < 3:
            raise ValueError(f"主题 '{subject_key}' 的数据必须至少有3列(X,Y,Z坐标)")
        
        # 收集其他数据集名称
        other_datasets = []
        for key in data_group.keys():
            if key not in ['r_glove_pos', 'l_glove_pos', 'frame_ids']:
                other_datasets.append(key)
        
        # 读取其他数据
        other_data = {}
        for key in other_datasets:
            other_data[key] = data_group[key][:]
        
        # 对右手数据进行坐标变换 z·=-x, y·=z, x·=-y
        r_aligned_data = apply_coordinate_transformation(r_glove_pos_data)
        
        # 对左手数据进行同样的坐标变换（如果存在）
        if has_left_hand:
            l_aligned_data = apply_coordinate_transformation(l_glove_pos_data)
        else:
            l_aligned_data = l_glove_pos_data
    
    # 如果提供了输出文件路径，则保存变换后的数据
    if output_file_path:
        with h5py.File(output_file_path, 'w') as out_h5_file:
            # 创建主题组
            group = out_h5_file.create_group(subject_key)
            
            # 将其他数据复制到新文件
            for key, value in other_data.items():
                group.create_dataset(key, data=value)
            
            # 将变换后的数据写入新文件
            group.create_dataset('r_glove_pos', data=r_aligned_data)
            group.create_dataset('l_glove_pos', data=l_aligned_data)
            group.create_dataset('frame_ids', data=frame_ids)
        
        print(f"已将主题 '{subject_key}' 坐标系对齐后的数据保存到: {output_file_path}")
    
    return r_aligned_data, l_aligned_data

def apply_coordinate_transformation(data):
    """
    应用坐标变换: z·=-x, y·=z, x·=-y
    """
    # 保存原始坐标
    original_x = data[..., 0].copy()  # x
    original_y = data[..., 1].copy()  # y
    original_z = data[..., 2].copy()  # z
    # 创建副本以避免修改原数据
    transformed_data = data.copy()
    # 应用变换1: z·=-x, y·=z, x·=-y
    transformed_data[..., 0] = -original_y  # x· = -y
    transformed_data[..., 1] = original_z  # y· = z
    transformed_data[..., 2] = -original_x  # z· = -x

    # 应用变换2: z·=x, y·=-z, x·=-y
    # transformed_data[..., 0] = -original_y  # x· = -y
    # transformed_data[..., 1] = -original_z  # y· = -z
    # transformed_data[..., 2] = original_x  # z· = x

    # 应用变换3: z·=-x, y·=-z, x·=-y
    # transformed_data[..., 0] = -original_y  # x· = -y
    # transformed_data[..., 1] = -original_z  # y· = -z
    # transformed_data[..., 2] = -original_x  # z· = -x
    
    return transformed_data

def align_all_subjects(file_path, output_file_path):
    """
    对H5文件中所有主题（一级key）执行坐标系对齐操作
    
    参数:
    file_path: 输入H5文件路径
    output_file_path: 输出H5文件路径
    """
    # 获取所有主题键
    with h5py.File(file_path, 'r') as h5_file:
        all_subjects = list(h5_file.keys())
    
    print(f"找到 {len(all_subjects)} 个主题: {all_subjects}")
    
    # 创建新的H5文件
    with h5py.File(output_file_path, 'w') as out_h5_file:
        for subject in all_subjects:
            print(f"正在处理主题: {subject}")
            
            # 从原文件读取数据
            with h5py.File(file_path, 'r') as h5_file:
                data_group = h5_file[subject]
                
                # 检查所需数据集是否存在
                if 'r_glove_pos' not in data_group:
                    print(f"跳过主题 '{subject}': 缺少 'r_glove_pos' 数据集")
                    continue
                
                # 读取r_glove_pos数据
                r_glove_pos_data = data_group['r_glove_pos'][:]
                
                # 检查是否有l_glove_pos数据
                has_left_hand = 'l_glove_pos' in data_group
                if has_left_hand:
                    l_glove_pos_data = data_group['l_glove_pos'][:]
                else:
                    # 如果没有左手数据，创建零矩阵
                    l_glove_pos_data = np.zeros_like(r_glove_pos_data)
                
                # 检查是否有frame_ids
                has_frame_ids = 'frame_ids' in data_group
                if has_frame_ids:
                    frame_ids = data_group['frame_ids'][:]
                else:
                    # 如果没有frame_ids，生成默认的frame_ids
                    frame_ids = np.arange(r_glove_pos_data.shape[0]).reshape(-1, 1)
                
                # 检查数据维度是否适合进行坐标变换
                if r_glove_pos_data.ndim < 2 or r_glove_pos_data.shape[-1] < 3:
                    print(f"跳过主题 '{subject}': 数据维度不符合要求")
                    continue
                
                # 收集其他数据集名称
                other_datasets = []
                for key in data_group.keys():
                    if key not in ['r_glove_pos', 'l_glove_pos', 'frame_ids']:
                        other_datasets.append(key)
                
                # 读取其他数据
                other_data = {}
                for key in other_datasets:
                    other_data[key] = data_group[key][:]
                
                # 对右手数据进行坐标变换
                r_aligned_data = apply_coordinate_transformation(r_glove_pos_data)
                
                # 对左手数据进行同样的坐标变换（如果存在）
                if has_left_hand:
                    l_aligned_data = apply_coordinate_transformation(l_glove_pos_data)
                else:
                    l_aligned_data = l_glove_pos_data
            
            # 创建主题组并保存数据
            group = out_h5_file.create_group(subject)
            
            # 将其他数据复制到新文件
            for key, value in other_data.items():
                group.create_dataset(key, data=value)
            
            # 将变换后的数据写入新文件
            group.create_dataset('r_glove_pos', data=r_aligned_data)
            group.create_dataset('l_glove_pos', data=l_aligned_data)
            group.create_dataset('frame_ids', data=frame_ids)
            
            print(f"已处理并保存主题: {subject}, 数据形状: {r_aligned_data.shape}")
    
    print(f"已完成所有主题的坐标系对齐，结果保存到: {output_file_path}")

# 使用示例
if __name__ == '__main__':
    # 文件路径配置
    h5_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\test_glove_data0204.h5'
    output_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\test_glove_data0204_aligned.h5'
    
    # 对所有主题执行坐标系对齐
    align_all_subjects(h5_file_path, output_file_path)
    
    # 验证结果
    print("\n验证结果:")
    with h5py.File(output_file_path, 'r') as h5_file:
        for subject in h5_file.keys():
            r_data = h5_file[subject]['r_glove_pos'][:]
            l_data = h5_file[subject]['l_glove_pos'][:]
            print(f"主题 '{subject}': r_glove_pos形状={r_data.shape}, l_glove_pos形状={l_data.shape}")