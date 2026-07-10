import h5py
import numpy as np

def split_datasets(input_file, test_output_file, train_output_file, test_frames=2400):
    """
    读取h5文件中的left_hand和right_hand数据集，
    将前1000帧作为测试数据，其余作为训练数据分别保存
    
    Args:
        input_file (str): 输入h5文件路径
        test_output_file (str): 测试数据输出文件路径
        train_output_file (str): 训练数据输出文件路径
        test_frames (int): 用作测试的帧数，默认为1000
    """
    key = '测试-ceshi'
    try:
        # 打开输入文件进行读取
        with h5py.File(input_file, 'r') as infile:
            # 处理测试数据文件
            with h5py.File(test_output_file, 'w') as test_outfile:
                # 创建名为"测试-ceshi"的组
                test_group = test_outfile.create_group(key)
                
                # 检查源数据集是否存在
                if 'left_hand' in infile:
                    # 读取left_hand前test_frames帧数据并重命名为l_glove_pos，保存在组内
                    left_data = infile['left_hand'][:test_frames]
                    test_group.create_dataset('l_glove_pos', data=left_data)
                    print(f"测试数据: 成功复制 left_hand -> {key}/l_glove_pos, 形状: {left_data.shape}")
                else:
                    print("警告: 源文件中未找到 'left_hand' 数据集")
                
                if 'right_hand' in infile:
                    # 读取right_hand前test_frames帧数据并重命名为r_glove_pos，保存在组内
                    right_data = infile['right_hand'][:test_frames]
                    test_group.create_dataset('r_glove_pos', data=right_data)
                    print(f"测试数据: 成功复制 right_hand -> {key}/r_glove_pos, 形状: {right_data.shape}")
                else:
                    print("警告: 源文件中未找到 'right_hand' 数据集")
                
                # 处理frame_ids（如果存在）
                if 'frame_ids' in infile:
                    frame_data = infile['frame_ids'][:test_frames]
                    test_group.create_dataset('frame_ids', data=frame_data)
                    print(f"测试数据: 复制 frame_ids, 形状: {frame_data.shape}")
            
            # 处理训练数据文件
            with h5py.File(train_output_file, 'w') as train_outfile:
                # 创建名为"测试-ceshi"的组
                train_group = train_outfile.create_group(key)
                
                # 检查源数据集是否存在
                if 'left_hand' in infile:
                    # 读取left_hand剩余帧数据并重命名为l_glove_pos，保存在组内
                    left_data = infile['left_hand'][test_frames:]
                    train_group.create_dataset('l_glove_pos', data=left_data)
                    print(f"训练数据: 成功复制 left_hand -> {key}/l_glove_pos, 形状: {left_data.shape}")
                else:
                    print("警告: 源文件中未找到 'left_hand' 数据集")
                
                if 'right_hand' in infile:
                    # 读取right_hand剩余帧数据并重命名为r_glove_pos，保存在组内
                    right_data = infile['right_hand'][test_frames:]
                    train_group.create_dataset('r_glove_pos', data=right_data)
                    print(f"训练数据: 成功复制 right_hand -> {key}/r_glove_pos, 形状: {right_data.shape}")
                else:
                    print("警告: 源文件中未找到 'right_hand' 数据集")
                
                # 处理frame_ids（如果存在）
                if 'frame_ids' in infile:
                    frame_data = infile['frame_ids'][test_frames:]
                    train_group.create_dataset('frame_ids', data=frame_data)
                    print(f"训练数据: 复制 frame_ids, 形状: {frame_data.shape}")
        
        print(f"数据分割完成:")
        print(f"  测试数据 ({test_frames}帧): {input_file} -> {test_output_file}")
        print(f"  训练数据 ({'剩余' if 'left_hand' in infile else '帧'}): {input_file} -> {train_output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file}'")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

def main():
    # 设置输入和输出文件路径
    input_file = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data_collect\\hand_data_0204.h5"           # 原始数据文件
    test_output_file = "test_glove_data.h5"    # 测试数据输出文件
    train_output_file = "train_glove_data.h5"  # 训练数据输出文件
    
    # 执行数据集分割操作
    split_datasets(input_file, test_output_file, train_output_file)

if __name__ == "__main__":
    main()