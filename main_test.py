# tensorboard --logdir D:\2026\code\TransHandR\TransHandR\checkpoint\logs\linker\experiment_20260204_211955

# tensorboard --logdir D:\2026\code\TransHandR\TransHandR\checkpoint\logs\experiment_20260129_194546


import h5py
import torch
import torch.nn as nn
from model.model_poseformer import PoseTransformer
from model.loss import RealTimeVisualizer, hand_loss
import torch.nn.functional as F
import time
import numpy as np
import os
from dataset.custom_dataset import linkhand_onehand_dataset            
# 创建碰撞损失函数
from model.loss import CollisionLoss , RegLoss
from model.angle2real import create_hand_kinematics
from config.variables_define import *
from dataset.generators import ChunkedGenerator
# batch_size = 1
def test_model_loading(model_path=None, input_path=None, output_file=None, keys=None, batch_size=1024):
    """
    测试加载指定路径的模型
    """
    if model_path is None:
        return KeyError("Model path must be provided.")
   
    dataset = linkhand_onehand_dataset(input_path, data_hand_type='right', scaling_factor=scaling_factor)
    
    # 如果keys为None，使用所有可用的keys
    if keys is None:
        keys = dataset.all_keys()
        
    hand_data = dataset.fetch(keys, action_filter=None, subset=1, parse_3d_poses=True, downsample=1)
    
    #输出文件不存在则创建
    if output_file is None:
        output_file = "output.h5"
        # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    if not os.path.exists(output_file):
        with h5py.File(output_file, 'w') as f:
            pass
    
    # 用于存储输出结果和时间
    all_outputs = []
    total_inference_time = 0.0
    frame_count = 0
    
    try:
        # 创建模型实例
        model = PoseTransformer(
        num_frame=receptive_field,
        in_num_joints=num_joints, 
        in_chans=3, 
        out_num_joint=out_num_joint, 
        out_chans=1, 
        embed_dim_ratio=embed_dim_ratio,
        spatial_depth=spatial_depth,     # 增加空间Transformer层数
        temporal_depth=temporal_depth,    # 减少时序Transformer层数
        spatial_mlp_ratio=spatial_mlp_ratio, # 增大空间MLP比例
        temporal_mlp_ratio=temporal_mlp_ratio, # 减小时序MLP比例
        num_heads=num_heads, 
        qkv_bias=qkv_bias, 
        qk_scale=None,
        drop_path_rate=drop_path_rate,
        angle_limit_rad=angle_limit_rob)
        
        # 如果有可用的GPU，将模型移到GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        hand_fk = create_hand_kinematics(urdf_file, hand_cfg, device, scale_factor=scaling_factor_rb)

        # 加载模型权重部分修改为：
        print(f"Loading model from: {model_path}")
        # 修改后的代码
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_pos'])
        print("Model loaded successfully!")
        print(f"Model is on device: {next(model.parameters()).device}")
        
        # 创建可视化器
        # visualizer = RealTimeVisualizer()
        visualizer = None
        
        # 测试模型推理
        print("Testing model inference...")
        
        model.eval()
        
        # # 定义损失函数
        # pos_loss_function = nn.MSELoss()
        # # 此处用余弦损失
        # vec_loss_function = nn.MSELoss()
        # # 碰撞损失网络出初始化
        # # 排除0号点，并且不计算(2,3)和(5,6)这对的碰撞
        # col_loss_function = CollisionLoss(
        #     threshold=0.03, 
        #     rb_dic=rb_dic,
        #     excluded_points=[0],
        #     excluded_pairs=[(1, 2), (4, 5), (7, 8), (10, 11), (14, 15)]
        # )
        # # regularization loss
        # reg_loss_function = RegLoss()
        
        # 创建ChunkedGenerator用于批量处理数据
        pad = (receptive_field - 1) // 2 # number_of_frames通过 pad 参数影响数据生成器
        # 准备约束条件
        # lower_bounds = torch.tensor([lim[0] for lim in angle_limit_rob], dtype=torch.float32)
        # upper_bounds = torch.tensor([lim[1] for lim in angle_limit_rob], dtype=torch.float32)
        causal_shift = 0 # 作用：表示因果偏移量，用于控制时间序列处理中的因果关系 值为0的含义：表示不进行时间上的偏移，即当前帧的输出只依赖于当前帧及之前的帧，不依赖未来帧
        stride = 1  # 每次移动一帧
        
        # 构建测试生成器
        test_generator = ChunkedGenerator(
            batch_size=batch_size,
            cameras=None, 
            poses_3d=hand_data,  # 使用原始数据构建生成器
            poses_2d=None, 
            chunk_length=stride,
            pad=pad, 
            causal_shift=causal_shift, 
            shuffle=False,  # 不打乱顺序
            augment=False,  # 不进行数据增强
            kps_left=None, 
            kps_right=None, 
            joints_left=None, 
            joints_right=None
        )
        
        print(f"INFO: Testing on {test_generator.num_frames()} frames")
        
        # 遍历生成器进行推理
        with torch.no_grad():
            for batch_idx, (cameras_batch, poses_3d_batch, poses_3d_padded_batch) in enumerate(test_generator.next_epoch()):
                # 将数据转换为tensor并移到设备上
                poses_3d_tensor = torch.from_numpy(poses_3d_padded_batch.astype('float32')).to(device)
                
                time0 = time.time()
                
                # 执行模型推理
                batch_outputs = model(poses_3d_tensor)
                # 将输出添加到结果列表
                all_outputs.extend([output.cpu().numpy() for output in batch_outputs])
                # 补充5列为0以匹配关节角度数量
                batch_outputs_padded = F.pad(batch_outputs, (0, 5, 0, 0), "constant", 0)
                
                
                
                time1 = time.time()
                time_used = time1 - time0
                total_inference_time += time_used
                frame_count += batch_outputs_padded.shape[0]  # 增加当前批次的帧数
                
                # 每处理100个批次输出进度
                if batch_idx % 100 == 0:
                    print(f"Processed batch {batch_idx}, current total frames: {frame_count}")
                    
                # 损失信息打印
                # print("损失形状:", loss_3d_pos.shape)
                # print("是否需要梯度:", loss_3d_pos.requires_grad)
                # print("损失值:", loss_3d_pos.item())
                # 在训练循环中，每次计算完损失后添加
                # 检查模型参数信息
                # print("模型参数梯度状态:")
                # for name, param in model_pos_train.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.shape}")
                #     else:
                #         print(f"{name}: No grad")
                # break
                
                # 使用可视化器进行损失计算和可视化
                # 创建一些虚拟的数据用于测试可视化功能
                dummy_source_3D = poses_3d_tensor[:, -1:, :, :]  # 取最后一帧，但保持维度存在
                # print(f"Computing loss with visualization for frame {frame_id}...")
                # loss, loss_1, loss_2, loss_3, loss_4, loss_5,loss_6 = hand_loss(
                #     batch_outputs_padded, 
                #     dummy_source_3D, 
                #     rb_dic, 
                #     source_dic, 
                #     pos_loss_function, 
                #     vec_loss_function, 
                #     col_loss_function, 
                #     reg_loss_function,
                #     visualizer=visualizer,
                #     hand_fk_model=hand_fk,
                #     logger=None,
                #     loss_weight=loss_weight
                # )
        
        print(f"Completed inference on all {frame_count} frames")
        
        # 将所有输出保存到HDF5文件
        with h5py.File(output_file, 'w') as f:
            # 将列表转换为numpy数组并保存
            outputs_array = np.array(all_outputs)
            f.create_dataset('outputs', data=outputs_array)
            f.attrs['num_frames'] = len(all_outputs)
            f.attrs['output_shape'] = outputs_array.shape[1:] if len(outputs_array.shape) > 1 else ()
        
        print(f"Outputs saved to {output_file}")
        
        # 计算并打印每帧平均用时
        if frame_count > 0:
            avg_time_per_frame = total_inference_time / frame_count
            print(f"Total inference time: {total_inference_time:.4f}s")
            print(f"Number of frames processed: {frame_count}")
            print(f"Average time per frame: {avg_time_per_frame:.6f}s ({1/avg_time_per_frame:.2f} FPS)")
        
        return model, visualizer
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None, None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# 消融实验类型
# ab_experiment_name = 'None'
# ab_experiment_name = 'ab_vec_loss'
# ab_experiment_name = 'ab_tip_pos_loss'
# ab_experiment_name = 'ab_col_loss'
# ab_experiment_name = 'ab_tip_dis_loss'
# ab_experiment_name = 'thumb'
# ab_experiment_name = 'thumb2'
ab_experiment_name = 'thumb3'
if __name__ == "__main__":
    if data_tpye == 'visionpro':
        input_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\glove_data_aligned.h5'
        # out_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_outputVP.h5'
        # input_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\test_glove_data0204_aligned.h5'
        # out_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output.h5'
        out_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_output.h5'
        # model_path=f'D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\{data_tpye}\\{hand_brand}\\model_final.pth'
        model_path=f'D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\{ab_experiment_name}\\{hand_brand}\\model_final.pth'
        # model_path=f'D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\None\\{hand_brand}\\model_final.pth'
        keys = None
    else:
        input_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove_aligned.h5'
        out_dir = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output.h5'
        model_path=f'D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\slahmr\\{hand_brand}\\model_final.pth'
        keys = ["S000079_P0008"]  # 选择特定的subject_key进行测试，或者设置为None使用所有数据
    
    model, visualizer = test_model_loading(
        input_path=input_dir,
        model_path=model_path,
        output_file=out_dir,
        keys=keys,
        batch_size=1024)
    
    if model:
        print("Model loading test completed successfully.")
    else:
        print("Model loading test failed.")