# 随机划分一个h5里面所有的键值，按7:1划分训练集和测试集，并输出键值到txt文件
import h5py
import numpy as np

def split_h5_keys_to_txt(h5_file_path, train_ratio=0.7, random_seed=42):
    """
    随机划分H5文件中的键到训练集和测试集，并保存到txt文件
    
    参数:
    h5_file_path: H5文件路径
    train_ratio: 训练集比例，默认0.7
    random_seed: 随机种子，默认42
    """
    # 设置随机种子以保证结果可复现
    np.random.seed(random_seed)
    
    # 读取H5文件中的所有键
    with h5py.File(h5_file_path, 'r') as f:
        all_keys = list(f.keys())
    
    # 打乱键的顺序
    shuffled_keys = np.random.permutation(all_keys)
    
    # 计算训练集大小
    num_train = int(len(shuffled_keys) * train_ratio)
    
    # 分割训练集和测试集
    train_keys = shuffled_keys[:num_train].tolist()  # 转换为Python列表
    test_keys = shuffled_keys[num_train:].tolist()   # 转换为Python列表
    
    # 输出统计信息
    print(f"H5文件中共有 {len(all_keys)} 个键")
    print(f"训练集: {len(train_keys)} 个键 ({train_ratio*100:.1f}%)")
    print(f"测试集: {len(test_keys)} 个键 ({(1-train_ratio)*100:.1f}%)")
    
    # 将训练集键保存到txt文件，以Python列表格式
    train_output_file = h5_file_path.replace('.h5', '_train_keys.txt')
    with open(train_output_file, 'w', encoding='utf-8') as f:
        f.write("# 训练集键列表\n")
        f.write("train_keys = [\n")
        for i, key in enumerate(train_keys):
            if i < len(train_keys) - 1:
                f.write(f"    '{key}',\n")
            else:
                f.write(f"    '{key}'\n")  # 最后一项不加逗号
        f.write("]\n")
    
    # 将测试集键保存到txt文件，以Python列表格式
    test_output_file = h5_file_path.replace('.h5', '_test_keys.txt')
    with open(test_output_file, 'w', encoding='utf-8') as f:
        f.write("# 测试集键列表\n")
        f.write("test_keys = [\n")
        for i, key in enumerate(test_keys):
            if i < len(test_keys) - 1:
                f.write(f"    '{key}',\n")
            else:
                f.write(f"    '{key}'\n")  # 最后一项不加逗号
        f.write("]\n")
    
    print(f"训练集键已保存到: {train_output_file}")
    print(f"测试集键已保存到: {test_output_file}")
    
    return train_keys, test_keys

if __name__ == "__main__":
    # 示例用法
    h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove_aligned.h5"  # 替换为实际的H5文件路径
    train_keys, test_keys = split_h5_keys_to_txt(h5_file_path)
    
    # 直接使用返回的Python列表
    print("训练集前5个键:", train_keys[:5])
    print("测试集前5个键:", test_keys[:5])