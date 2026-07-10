import torch
import torch.nn as nn
from model.model_poseformer import PoseTransformer
# Import only the variables we need to avoid wildcard import pollution
from config.variables_define import (
    receptive_field,
    num_joints,
    out_num_joint,
    embed_dim_ratio,
    spatial_depth,
    temporal_depth,
    spatial_mlp_ratio,
    temporal_mlp_ratio,
    num_heads,
    drop_path_rate,
    angle_limit_rob,
)
def check_model_parameters(model):
    """
    检查模型参数是否有异常
    """
    print("\n=== 模型参数检查 ===")
    
    total_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0
    large_params = 0
    small_params = 0
    
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        # 如果没有参数（极少见），跳过
        if numel == 0:
            print(f"Layer: {name} has no parameters, skipping stats")
            continue

        # 将参数移到 cpu 并以 numpy 进行统计，避免 GPU/grad 相关问题
        pdata = param.detach().cpu()

        # 检查参数的统计信息
        param_mean = pdata.mean().item()
        # 使用无偏差修正以避免在某些较小张量上触发 degrees of freedom 警告
        try:
            param_std = pdata.std(unbiased=False).item()
        except TypeError:
            # 兼容旧版 PyTorch
            param_std = pdata.std().item()
        param_min = pdata.min().item()
        param_max = pdata.max().item()

        # 检查异常值
        nan_count = torch.isnan(pdata).sum().item()
        inf_count = torch.isinf(pdata).sum().item()
        zero_count = (pdata == 0).sum().item()
        large_count = (torch.abs(pdata) > 100).sum().item()
        small_count = (torch.abs(pdata) < 1e-8).sum().item()
        
        nan_params += nan_count
        inf_params += inf_count
        zero_params += zero_count
        large_params += large_count
        small_params += small_count
        
        print(f"Layer: {name}")
        print(f"  Shape: {list(param.shape)}, Params: {param.numel()}")
        print(f"  Mean: {param_mean:.6f}, Std: {param_std:.6f}")
        print(f"  Min: {param_min:.6f}, Max: {param_max:.6f}")
        print(f"  NaN: {nan_count}, Inf: {inf_count}, Zero: {zero_count}")
        print(f"  Large(>100): {large_count}, Small(<1e-8): {small_count}")
        print()
    
    print(f"=== 总体参数统计 ===")
    print(f"总参数数量: {total_params}")
    if total_params == 0:
        print("模型没有参数，无法计算统计信息")
        return False

    def pct(x):
        return x/total_params*100

    print(f"零参数数量: {zero_params} ({pct(zero_params):.2f}%)")
    print(f"NaN参数数量: {nan_params} ({pct(nan_params):.2f}%)")
    print(f"Infinity参数数量: {inf_params} ({pct(inf_params):.2f}%)")
    print(f"大参数数量(>100): {large_params} ({pct(large_params):.2f}%)")
    print(f"小参数数量(<1e-8): {small_params} ({pct(small_params):.2f}%)")
    
    # 检查是否存在异常
    has_issues = False
    if nan_params > 0:
        print("[WARN] 发现 NaN 参数!")
        has_issues = True
    if inf_params > 0:
        print("[WARN] 发现 Infinity 参数!")
        has_issues = True
    if pct(large_params) > 1.0:  # 超过1%
        print("[WARN] 发现过多的大参数值!")
        has_issues = True
    if pct(zero_params) > 50.0:  # 超过50%
        print("[WARN] 发现过多的零参数值!")
        has_issues = True
    
    if not has_issues:
        print("[OK] 模型参数检查通过，未发现明显异常")
    
    return not has_issues

def check_model_loading(model_path=None):
    """
    检查加载的模型参数
    """
    if model_path is None:
        model_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\测试用\\model_final.pth"
    
    # 定义模型参数
    
    def _clean_state_dict(state_dict):
        """Strip 'module.' prefix if present (DataParallel checkpoints)."""
        new_state = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            new_state[new_key] = v
        return new_state

    try:
        # 创建模型实例
        model = PoseTransformer(
            num_frame=receptive_field, 
            in_num_joints=num_joints, 
            in_chans=3, 
            out_num_joint=out_num_joint, 
            out_chans=1,
            embed_dim_ratio=embed_dim_ratio, 
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            spatial_mlp_ratio=spatial_mlp_ratio,
            temporal_mlp_ratio=temporal_mlp_ratio, 
            num_heads=num_heads,
            qkv_bias=True, 
            qk_scale=None,
            drop_path_rate=drop_path_rate,
            angle_limit_rad=angle_limit_rob
        )

        # 如果有可用的GPU，将模型移到GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 如果使用了DataParallel，在加载权重之前需要先转换回来
        if isinstance(model, nn.DataParallel):
            model = model.module

        # 加载模型权重
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # 支持多种 checkpoint 格式
        if isinstance(checkpoint, dict):
            # 优先尝试常见键
            if 'model_pos' in checkpoint:
                state = checkpoint['model_pos']
            elif 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state = checkpoint['model']
            else:
                # 可能直接就是 state_dict，但还在 dict 中
                state = checkpoint
        else:
            state = checkpoint

        # 如果 key 含有 module. 前缀，去掉以兼容单/多卡
        if isinstance(state, dict):
            state = _clean_state_dict(state)

        model.load_state_dict(state, strict=False)
        
        print("Model loaded successfully!")
        # 如果模型有参数，打印参数所在设备；否则说明模型为空
        try:
            first_param = next(model.parameters())
            print(f"Model is on device: {first_param.device}")
        except StopIteration:
            print("Model has no parameters.")
        
        # 检查模型参数
        params_ok = check_model_parameters(model)
        
        return model, params_ok
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None, False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser(description="检查并加载模型参数")
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型文件路径 (.pth)。若不提供，脚本会在 checkpoint/models 下搜索 .pth 文件并选择第一个。")
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        # 在 checkpoint/models 下搜索 .pth 文件
        search_root = os.path.join(os.path.dirname(__file__), 'checkpoint', 'models')
        pattern = os.path.join(search_root, '**', '*.pth')
        found = glob.glob(pattern, recursive=True)
        if found:
            print('在 checkpoint/models 下找到以下模型文件：')
            for f in found:
                print('  ', f)
            model_path = found[0]
            print('未提供 --model-path，选择第一个模型文件：', model_path)
        else:
            default_path = os.path.join(os.path.dirname(__file__), 'checkpoint', 'models', '测试用', 'model_final.pth')
            print('未找到任何 .pth 模型文件，尝试默认路径：', default_path)
            model_path = default_path

    model, params_ok = check_model_loading(model_path=model_path)
    if model is not None:
        print(f"Model parameter check completed. Parameters OK: {params_ok}")
    else:
        print("Model parameter check failed.")