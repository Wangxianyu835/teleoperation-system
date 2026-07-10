# 数据结构可视化脚本 输入h5文件路径 输出数据结构
import h5py
import os
import sys

# python D:\code\TransHandR\TransHandR\dataset\data_collect\data_structure.py D:\code\TransHandR\TransHandR\dataset\data\glove_data.h5
def print_h5_structure(file_path):
    """
    读取h5文件并输出其结构树
    
    Args:
        file_path (str): h5文件路径
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件: {file_path}")
            print("=" * 50)
            print_structure(f, "")
    except Exception as e:
        print(f"读取文件时出错: {e}")

def print_structure(obj, prefix=""):
    """
    递归打印h5文件的结构
    
    Args:
        obj: h5py对象 (文件或组)
        prefix (str): 缩进前缀
    """
    if isinstance(obj, h5py.File):
        for key in obj.keys():
            item = obj[key]
            print_item(item, key, prefix)
    elif isinstance(obj, h5py.Group):
        for key in obj.keys():
            item = obj[key]
            print_item(item, key, prefix)

def print_item(item, key, prefix):
    """
    打印单个项目的信息
    
    Args:
        item: h5py对象
        key (str): 键名
        prefix (str): 缩进前缀
    """
    if isinstance(item, h5py.Group):
        print(f"{prefix}📁 {key}/")
        print_structure(item, prefix + "  ")
    elif isinstance(item, h5py.Dataset):
        shape = item.shape
        dtype = item.dtype
        print(f"{prefix}📄 {key} {shape} {dtype}")
    else:
        print(f"{prefix}❓ {key} (未知类型)")

if __name__ == "__main__":
    # 如果提供了命令行参数，则使用该参数作为文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 否则提示用户输入文件路径
        file_path = input("请输入h5文件路径: ").strip()
    
    print_h5_structure(file_path)