'''
修改自run_poseformer.py，用于训练Retransformer
'''

#代码注释：
'''
    基于给出的数据集训练一个模型，模型输入是3D关节位置，输出是手部关节的角度。训练过程中使用多个损失函数，包括3D位置损失、向量损失、碰撞损失等，
    以确保模型不仅能够准确预测关节角度，还能满足物理约束和手部结构的合理性。训练过程中还会记录损失值和梯度信息，以便分析训练过程中的问题，
    并使用TensorBoard进行可视化。最终训练完成后会保存模型权重，以供后续评估和使用。
'''



import torch.optim as optim#从pytorch导入优化器
import errno#导入错误号模块，用于处理文件系统相关的错误
from tensorboardX import SummaryWriter#用于记录训练过程中的指标和损失，以便在TensorBoard中可视化

#以下是项目中自定义的模块和函数
from config.arguments import parse_args#用于解析命令行参数，获取训练配置
from config.utils import *#一些通用函数，包括日志初始化，模型参数转换等
from config.variables_define import *#定义了一些全局变量和配置参数，如模型结构参数、损失权重、数据路径等
from dataset.generators import ChunkedGenerator, UnchunkedGenerator#用于生成训练和评估数据的迭代器，支持分块处理和不分块处理
from model.angle2real import create_hand_kinematics#用于创建手部运动学模型，基于URDF文件和手部配置参数
from model.model_poseformer import *#定义了PoseTransformer模型结构，基于Transformer架构设计，用于处理手部姿态估计任务
from model.loss import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # python main.py --dataset h5 --keypoints glove_data --checkpoint ./checkpoint
    # 解析参数
    args = parse_args()
    log =  logging.getLogger()
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)
    # 在创建checkpoint目录后，创建SummaryWriter
    # 创建带时间戳的实验名称
    # 消融实验类型
    # ab_experiment_name = 'None'
    # ab_experiment_name = 'ab_vec_loss'
    # ab_experiment_name = 'ab_tip_pos_loss'
    # ab_experiment_name = 'ab_col_loss'
    # ab_experiment_name = 'ab_tip_dis_loss'
    # ab_experiment_name = 'thumb'
    # ab_experiment_name = 'thumb2'
    ab_experiment_name = 'thumb3'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"
    if data_tpye == 'slahmr':
        log_dir = os.path.join(args.checkpoint, f'logs\\{data_tpye}\\{hand_brand}\\{experiment_name}')
    else:
        log_dir = os.path.join(args.checkpoint, f'logs\\{ab_experiment_name}\\{hand_brand}\\{experiment_name}')

    writer = SummaryWriter(log_dir=log_dir)
    t0_start = time.time()

    logger = setup_logging()
    logger.info(f"最终版调试{ab_experiment_name}，数据类型: {data_tpye}，手的类型: {hand_brand}")
    logger.info(f"补回指尖损失")
    # 保存模型文件
    if data_tpye == 'slahmr':
        model_save_path = f'D:\\2026\\code\\mytrans\\checkpoint\\models\\{data_tpye}\\{hand_brand}\\model_final.pth'
    else:
        model_save_path = f'D:\\2026\\code\\mytrans\\checkpoint\\models\\{ab_experiment_name}\\{hand_brand}\\model_final.pth'
    # logger.info(f"数据类型: {data_tpye}")
    logger.info(f"保存模型文件: {model_save_path}")
    # logger.info(f"手的类型: {hand_brand}")
    # 加载数据集
    print('Loading dataset...')
    hand_type = args.hand_type

    if args.dataset == 'h36m':
        print('Human3.6M dataset not supported')
    else:
        from dataset.custom_dataset import linkhand_onehand_dataset

        # dataset_path = 'dataset/data/' + args.dataset + '/' + args.keypoints + '.h5'
        dataset_path = 'dataset/data/' + args.dataset + '/' + keypoints + '.h5'
        print('windows路径格式注意:',dataset_path)
        dataset = linkhand_onehand_dataset(dataset_path, data_hand_type=hand_type, scaling_factor=scaling_factor)
    print('Dataset loaded successfully')

    # 准备数据
    print('Preparing data...')
    #检查数据帧标签和角度数据是否对齐,执行数据检查
    dataset.check_alignment()
    keys = dataset.all_keys()
    # 定义训练和测试主体
    if data_tpye == 'slahmr':
        subjects_train = train_keys
    else:
        subjects_train = keys
    # 选取动作
    action_filter = None
    poses_train = dataset.fetch(subjects_train, action_filter)

    # 定义训练和评估用的PoseTransformer模型
    #定义模型的感受野大小：表示模型在时间序列处理中能够"看到"的帧数范围，决定模型的时间上下文长度：控制模型处理当前帧时可以利用的历史帧和未来帧的数量
    print('INFO: Receptive field: {} frames'.format(receptive_field))

    pad = (receptive_field -1) // 2 # number_of_frames通过 pad 参数影响数据生成器
    # 准备约束条件
    # lower_bounds = torch.tensor([lim[0] for lim in angle_limit_rob], dtype=torch.float32)
    # upper_bounds = torch.tensor([lim[1] for lim in angle_limit_rob], dtype=torch.float32)
    # # 加载模型
    model_pos_train = PoseTransformer(num_frame=receptive_field, in_num_joints=num_joints, in_chans=in_chans, 
            out_num_joint=out_num_joint, out_chans=1, 
            embed_dim_ratio=embed_dim_ratio,
            spatial_depth=spatial_depth,     # 增加空间Transformer层数
            temporal_depth=temporal_depth,    # 减少时序Transformer层数
            spatial_mlp_ratio=spatial_mlp_ratio, # 增大空间MLP比例
            temporal_mlp_ratio=temporal_mlp_ratio, # 减小时序MLP比例
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            drop_path_rate=drop_path_rate,
            angle_limit_rad=angle_limit_rob)

    pos_loss = nn.MSELoss()
    # 此处用余弦损失
    vec_loss = nn.CosineEmbeddingLoss()
    vec_loss = None
    # 碰撞损失网络出初始化
    # 排除0号点，并且不计算(2,3)和(5,6)这对的碰撞
    col_loss = CollisionLoss(
        threshold=col_threshold, 
        rb_dic=rb_dic,
        excluded_points=[0],
        excluded_pairs=excluded_pairs
    )
    # regularization loss
    reg_criterion = RegLoss()
    if torch.cuda.is_available():
        # model_pos = nn.DataParallel(model_pos)
        # model_pos = model_pos.cuda()
        # model_pos_train = nn.DataParallel(model_pos_train)
        model_pos_train = model_pos_train.cuda()
    else:
        print('CUDA is not available, using CPU instead')
        assert False, "CPU mode is not supported"
    if logger is not None:
            logger.info(f'loss_weight: {loss_weight}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_fk = create_hand_kinematics(urdf_file, hand_cfg, device, scale_factor=scaling_factor_rb,axis_correction_matrix=correction_matrix)

    #################
    causal_shift = 0 # 作用：表示因果偏移量，用于控制时间序列处理中的因果关系 值为0的含义：表示不进行时间上的偏移，即当前帧的输出只依赖于当前帧及之前的帧，不依赖未来帧
    model_params = 0 # 作用：用于统计模型的总参数量 计算方式：通过遍历模型的所有参数（model_pos.parameters()），累加每个参数张量的元素数量（parameter.numel()）
    for parameter in model_pos_train.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    t1_train_start = time.time()
    ###################

    # 创建可视化器
    # visualizer = RealTimeVisualizer(actual_connections=hand_connections, robot_connections=robot_connections)
    visualizer = None
    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.01,eps=1e-6)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    # batch_size为 1024, stride为 1, pad为 pad = (receptive_field -1) // 2即 (3-1)//2= 1 casual_shift=0
    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras=None, poses_3d = poses_train, poses_2d=None, chunk_length=args.stride,
                                        pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                        kps_left=None, kps_right=None, joints_left=None, joints_right=None)
    print('INFO: Training on {} frames'.format(train_generator.num_frames()))


    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')
    # return 0

    while epoch < args.epochs:
        # start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
    #         本项目输入只输入3D,不需要这个loss epoch_loss_2d_train_unlabeled = 0
        model_pos_train.train()
        batch_idx = 0
        for cameras_train, batch_3D_single_frame, batch_3d_pad in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d_pad.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            optimizer.zero_grad()
            single_frame_3d =torch.from_numpy(batch_3D_single_frame.astype('float32'))
            if torch.cuda.is_available():
                single_frame_3d = single_frame_3d.cuda()
            # Predict angle
            # print("模型输入格式",inputs_3d.shape)
            batch_size = inputs_3d.shape[0]
            predicted_angle = model_pos_train(inputs_3d)
            # print("模型输出格式",predicted_angle.shape)
            # (frame, joint_num)
            # 为predicted_angle加五列零 以补足指尖固定关节的角度
            predicted_angle_padded = F.pad(predicted_angle, (0, 5, 0, 0), "constant", 0)
            # print("补足后模型输出格式",predicted_angle_padded.shape)
            # 对首维取平均值
            predicted_angle_mean = predicted_angle_padded.mean(dim=0)
            # print("模型输出",predicted_angle_mean)
            del inputs_3d
            torch.cuda.empty_cache()
            loss_total,loss_3d_vec,loss_3d_pos,loss_collision,loss_thumb,loss_tip_distance, thumb_loss2 = hand_loss(predicted_angle_padded, 
            single_frame_3d,rb_dic,source_dic,pos_loss, vec_loss, col_loss, reg_criterion,visualizer,hand_fk,logger,loss_weight)
            # 更新可视化（在主循环中）
            # 处理GUI事件
            if visualizer is not None:  # 如果有可视化器
                visualizer.update_plot()  # 更新绘图
                plt.pause(0.001)  # 短暂停顿以允许GUI更新
            # print("损失函数输出格式",loss_3d_pos.shape)
            loss_total.backward()

            # 记录梯度信息
            if batch_idx % 10 == 0:  # 每10个batch记录一次
                total_grad_norm = 0
                param_count = 0
                
                grad_info = f"\n--- Batch {batch_idx}, Epoch {epoch} ---"
                grad_info += f"\nLoss values - Total: {loss_total.item():.6f}, Vec: {loss_3d_vec.item():.6f}, Pos: {loss_3d_pos.item():.6f}, Col: {loss_collision.item():.6f}, Thumb: {loss_thumb.item():.6f}, TipDist: {loss_tip_distance.item():.6f}, Thumb2: {thumb_loss2.item():.6f}"
                
                for name, param in model_pos_train.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm
                        param_count += 1
                        
                        # # 只记录梯度较大的参数
                        # if grad_norm > 0.001:
                        #     grad_info += f"\n  {name}: grad_norm = {grad_norm:.6f}"
                
                avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
                grad_info += f"\n  Average gradient norm: {avg_grad_norm:.6f}"
                grad_info += f"\n  Total parameters with gradients: {param_count}"
                
                logger.info(grad_info)
            
            # 梯度裁剪（可选，有助于稳定训练）
            torch.nn.utils.clip_grad_norm_(model_pos_train.parameters(), max_norm=10.0)
            optimizer.step()
            batch_idx += 1
            writer.add_scalar('Loss/train_total', loss_total.item(), epoch*train_generator.num_batches+batch_idx)
            writer.add_scalar('Loss/train_vec', loss_3d_vec.item(), epoch*train_generator.num_batches+batch_idx)
            writer.add_scalar('Loss/train_pos', loss_3d_pos.item(), epoch*train_generator.num_batches+batch_idx)
            writer.add_scalar('Loss/train_collision', loss_collision.item(), epoch*train_generator.num_batches+batch_idx)
            writer.add_scalar('Loss/train_thumb', loss_thumb.item(), epoch*train_generator.num_batches+batch_idx)
            writer.add_scalar('Loss/train_tip_distance', loss_tip_distance.item(), epoch*train_generator.num_batches+batch_idx)
            writer.add_scalar('Loss/train_thumb2', thumb_loss2.item(), epoch*train_generator.num_batches+batch_idx)
            '''
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
            '''

        epoch += 1
        print('epoch:', epoch)

    # 检查路径是否存在
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    # 在训练完成后保存模型时，使用.pth格式
    torch.save({
        'epoch': epoch,
        'model_pos': model_pos_train.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr': lr,
    }, model_save_path)
    # 在程序结束前关闭writer
    # 在程序结束前关闭可视化器
    t2_end = time.time()
    print('训练时间:', t2_end - t1_train_start)
    print('总时间:', t2_end - t0_start)
    if visualizer is not None:
        plt.ioff()  # 关闭交互模式
        plt.show()  # 显示最终图形
    writer.close()

if __name__ == '__main__':
    main()
