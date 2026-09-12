import h5py
import numpy as np
import os

def read_h5_all_keys(h5_file_path):
    """
    读取H5文件中的所有键名（数据集和组）
    
    Args:
        h5_file_path: H5文件路径
    
    Returns:
        list: 包含所有键名的列表
    """
    all_keys = []
    
    with h5py.File(h5_file_path, 'r') as file:
        def get_all_keys(name, obj):
            all_keys.append(name)
        
        file.visititems(get_all_keys)
    
    return all_keys

def truncate_first_level_keys(keys, max_length=15):
    """
    截断第一层键名，保留前N个字符
    
    Args:
        keys: 原始键名列表
        max_length: 最大长度，默认15
    
    Returns:
        list: 截断后的键名列表
    """
    truncated_keys = []
    
    for key in keys:
        # 分割路径
        parts = key.split('/', 1)  # 只分割第一个'/'，保留剩余部分
        
        if len(parts) > 1:
            # 如果有多个层级，只截断第一层
            first_level = parts[0][:max_length]
            remaining = parts[1]
            new_key = f"{first_level}/{remaining}"
        else:
            # 如果只有一层，直接截断
            new_key = key[:max_length]
        
        truncated_keys.append(new_key)
    
    return truncated_keys

def copy_h5_with_truncated_names_and_frame_ids(input_path, output_path, max_length=15):
    """
    复制H5文件并将第一层键名截断后保存为新文件，同时为每个组添加frame_ids
    
    Args:
        input_path: 输入H5文件路径
        output_path: 输出H5文件路径
        max_length: 第一层键名的最大长度
    """
    with h5py.File(input_path, 'r') as input_file:
        with h5py.File(output_path, 'w') as output_file:
            def process_item(name, obj):
                # 分割路径
                parts = name.split('/', 1)  # 只分割第一个'/'
                
                if len(parts) > 1:
                    # 如果有多个层级，只截断第一层
                    first_level = parts[0][:max_length]
                    remaining = parts[1]
                    new_name = f"{first_level}/{remaining}"
                else:
                    # 如果只有一层，直接截断
                    new_name = name[:max_length]
                
                # 如果是数据集，先复制
                if isinstance(obj, h5py.Dataset):
                    output_file[new_name] = obj[()]
                    # 复制属性
                    for attr_name, attr_value in obj.attrs.items():
                        output_file[new_name].attrs[attr_name] = attr_value
                elif isinstance(obj, h5py.Group):
                    # 创建新组
                    group = output_file.create_group(new_name)
                    # 复制组属性
                    for attr_name, attr_value in obj.attrs.items():
                        group.attrs[attr_name] = attr_value
            
            # 遍历并复制所有对象
            input_file.visititems(process_item)
            
            # 为每个组添加frame_ids
            def add_frame_ids_to_groups(name, obj):
                if isinstance(obj, h5py.Group):
                    # 查找当前组下的所有数据集，获取最大帧数
                    max_frames = 0
                    pos_datasets = []  # 存储位置数据集
                    
                    for key in obj.keys():
                        dataset = obj[key]
                        if isinstance(dataset, h5py.Dataset):
                            # 检查是否是位置数据（通常具有3个维度：帧数, 关节点数, 坐标）
                            if len(dataset.shape) >= 2:
                                frame_count = dataset.shape[0]
                                if frame_count > max_frames:
                                    max_frames = frame_count
                                pos_datasets.append((key, dataset))
                    
                    # 如果找到了数据集，创建frame_ids
                    if max_frames > 0:
                        frame_ids = np.arange(max_frames).reshape(-1, 1)  # 形状为 (num_frames, 1)
                        
                        # 添加frame_ids数据集到组中
                        obj.create_dataset('frame_ids', data=frame_ids, dtype='i4')
                        print(f"为组 '{name}' 添加了 {max_frames} 帧的 frame_ids")
            
            # 遍历新文件中的所有组，添加frame_ids
            output_file.visititems(add_frame_ids_to_groups)

# 使用示例
if __name__ == "__main__":
    input_h5_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\sign_glove.h5"
    output_h5_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\sign_glove_truncated.h5"
    
    # 读取原始键名
    keys = read_h5_all_keys(input_h5_path)
    # print("原始键名:", keys)

    
    # 显示截断后的键名
    truncated_keys = truncate_first_level_keys(keys, 13)
    # print("截断后的键名:", truncated_keys)

    
    # 复制H5文件并使用截断的键名，同时添加frame_ids
    copy_h5_with_truncated_names_and_frame_ids(input_h5_path, output_h5_path, 13)
    
    print(f"已将修改后的H5文件保存为: {output_h5_path}")
    
    # 验证新文件
    new_keys = read_h5_all_keys(output_h5_path)
    # print("新文件中的键名:", new_keys)
    
    # 检查新文件中的frame_ids
    with h5py.File(output_h5_path, 'r') as f:
        for key in f.keys():
            group = f[key]
            # if 'frame_ids' in group:
                # print(f"组 '{key}' 包含 frame_ids，形状: {group['frame_ids'].shape}")