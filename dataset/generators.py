# tensorboard --logdir D:\2026\code\TransHandR\TransHandR\checkpoint\logs\experiment_20260129_194546

from itertools import zip_longest
import numpy as np

'''
ChunkedGenerator（分块数据生成器）
- 训练数据生成器：用于训练阶段的数据批处理
- 数据分块处理：将连续的视频帧数据分割成固定大小的块（chunks）
- 支持数据增强：可以通过 augment 参数启用数据增强
- 批量处理：配合 args.batch_size 进行批量数据处理，提高训练效率

UnchunkedGenerator（非分块数据生成器）
- 评估数据生成器：用于模型评估和验证阶段
- 完整序列处理：不将数据分割成块，保持数据的时间连续性
- 无数据增强：通常设置 augment=False，确保评估的一致性
- 测试数据准备：为测试集提供连续的帧序列，用于准确的模型性能评估
'''

class ChunkedGenerator:
    """
    批量数据生成器，用于训练阶段
    序列被分割成等长的块并根据需要进行填充

    参数说明：
    batch_size -- 训练时使用的批次大小
    cameras -- 摄像机列表，每个视频一个元素（可选，用于半监督训练）
    poses_3d -- 3D姿态列表，每个视频一个元素（可选，用于监督训练）
    poses_2d -- 2D关键点输入列表，每个视频一个元素（在3D输入中不使用）
    chunk_length -- 每个训练样本要预测的帧数（通常为1）
    pad -- 2D输入填充，补偿有效卷积，每边（取决于感受野）
    causal_shift -- 使用因果卷积时的不对称填充偏移（通常为0或"pad"）
    shuffle -- 在每个epoch前随机打乱数据集
    random_seed -- 随机生成器的初始种子
    augment -- 通过水平翻转姿态进行数据增强
    kps_left 和 kps_right -- 如果启用翻转，则为左/右2D关键点列表
    joints_left 和 joints_right -- 如果启用翻转，则为左/右3D关节列表

    输出目标长度:每个chunk的目标输出包含 chunk_length 帧数据
    默认值为1:在代码中设置为1,意味着每次预测下一帧
    输入：[chunk_length + 2*pad] 帧（包含上下文信息）
    输出：[chunk_length] 帧（要预测的目标）
    分段依据：按 chunk_length 将长序列分割成小段
    边界定义：每个chunk覆盖 chunk_length 帧的时间跨度
    """
        # batch_size为 1024, chunklength为 1, pad为 pad = (receptive_field -1) // 2即 (3-1)//2= 1 casual_shift=0
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
             chunk_length, pad=0, causal_shift=0,
             shuffle=True, random_seed=1234,
             augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
             endless=False):
        """
        初始化ChunkedGenerator，针对3D输入数据的训练
        poses_2d 现在为 None，poses_3d 为实际输入数据
        """
        # 检查camera和poses_2d是否为None，如果不是None则报错
        assert cameras is None or poses_2d is None
        # 使用poses_3d作为输入数据源
        effective_poses = poses_3d
        # 输出数据格式
        print("poses_3d[0]:", poses_3d[0].shape)
        print("禁用2D输入数据和相机数据，只使用3D输入数据进行训练,数据生成器初始化")
        cameras = None
        poses_2d = None
        # 构建序列信息
        pairs = [] 
        # (seq_idx, start_frame, end_frame, flip) 元组
        # 数据索引映射
        # seq_idx: 指明数据来自哪个原始序列
        # start_frame, end_frame: 定义从原始序列中要提取的帧范围
        # flip: 指示是否需要翻转数据（当前代码中始终为False）
        for i in range(len(effective_poses)):
            # 对于3D输入数据进行处理
            # 计算需要多少个块来覆盖整个序列
            n_chunks = (effective_poses[i].shape[0] + chunk_length - 1) // chunk_length
            # 计算偏移量以平均分布填充 offset?
            offset = (n_chunks * chunk_length - effective_poses[i].shape[0]) // 2
            # 创建边界数组，定义每个块的起始和结束位置
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            # 创建增强向量（对于3D单手数据始终为False）
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            # 对于单手数据，我们不进行翻转增强，所以只添加原始序列对
            # zip将序列索引、起始边界、结束边界和翻转向量组合成元组
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            # 注意：由于是单手数据，禁用了翻转增强

        # 初始化批处理缓冲区 - 基于3D数据
        # 检查是否有3D姿态数据
        if poses_3d is not None:
            # 3D数据的输入缓冲区 (batch_size, chunk_length + 2*pad, num_joints, 3)
            # 用于网络输入的数据
            self.batch_3d_in = np.empty((batch_size, chunk_length + 2*pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
            # 用于评估的数据
            self.batch_out = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
            # print("网络输入数据格式",self.batch_3d_in.shape)
            # print("评估数据格式",self.batch_out.shape)
        else:
            # 如果没有3D姿态数据，抛出异常
            raise ValueError("poses_3d cannot be None for 3D input")

        # 计算批处理数量：总数据对数除以批次大小，向上取整
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        # 初始化随机数生成器，用于数据打乱
        self.random = np.random.RandomState(random_seed)
        # 存储数据对列表，用于索引原始数据
        self.pairs = pairs
        # 是否启用数据打乱
        self.shuffle = shuffle
        # 填充大小
        self.pad = pad
        # 因果偏移
        self.causal_shift = causal_shift
        # 是否无限循环
        self.endless = endless
        # 存储当前状态，用于恢复训练
        self.state = None
        
        # 由于是3D输入，重新定义数据源
        self.cameras = cameras  # 摄像机参数（已设为None）
        self.poses_3d = poses_3d  # 3D输入数据
        self.poses_2d = poses_2d  # 2D数据不使用（已设为None）
        
        # 对于单手数据，禁用翻转增强
        self.augment = False  # 强制禁用翻转
        self.kps_left = kps_left  # 左侧关键点索引
        self.kps_right = kps_right  # 右侧关键点索引
        self.joints_left = joints_left  # 左侧关节索引
        self.joints_right = joints_right  # 右侧关节索引
        
    def num_frames(self):
        """
        获取总帧数
        返回: 批处理数量 * 批处理大小
        """
        return self.num_batches * self.batch_size
    
    def random_state(self):
        """
        获取当前随机状态
        返回: numpy随机数生成器对象
        """
        return self.random
    
    def set_random_state(self, random):
        """
        设置随机状态
        参数: random - 要设置的随机数生成器对象
        """
        self.random = random
        
    def augment_enabled(self):
        """
        检查是否启用了数据增强
        返回: 布尔值，表示是否启用数据增强
        """
        return self.augment

    def next_pairs(self):
        """
        获取下一批数据对
        返回: (起始索引, 数据对列表)
        """
        if self.state is None:
            # 如果没有状态，重新洗牌或使用原始顺序
            if self.shuffle:
                # 随机打乱数据对顺序
                pairs = self.random.permutation(self.pairs)
            else:
                # 使用原始顺序
                pairs = self.pairs
            return 0, pairs
        else:
            # 如果有状态，返回当前状态
            return self.state
    
    def next_epoch(self):
        """
        获取下一个epoch的数据生成器，针对3D输入数据的训练
        生成器函数，逐批返回训练数据
        """
        enabled = True  # 控制循环的标志
        while enabled:
            # 获取下一批数据对
            start_idx, pairs = self.next_pairs()
            # 遍历所有批次
            for b_i in range(start_idx, self.num_batches):
                # 获取当前批次的数据对
                # 即部分数据对
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]

                # 遍历批次中的每个数据样本
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    # 计算3D姿态的起始和结束位置（考虑填充和因果偏移）
                    start_pose = start_3d - self.pad - self.causal_shift
                    end_pose = end_3d + self.pad - self.causal_shift

                    # 处理3D输入数据
                    seq_pose = self.poses_3d[seq_i]  # 获取当前序列的3D姿态数据
                    # 计算有效范围的边界
                    low_pose = max(start_pose, 0)  # 确保不低于0
                    high_pose = min(end_pose, seq_pose.shape[0])  # 确保不超过序列长度
                    # 计算需要填充的边界大小
                    pad_left_pose = low_pose - start_pose  # 左侧需要填充的帧数
                    pad_right_pose = end_pose - high_pose  # 右侧需要填充的帧数
                    
                    # 获取当前片段的数据
                    current_segment = seq_pose[low_pose:high_pose]
                    
                    # 处理3D输入数据：应用填充到输入数据,补充至感受野长度
                    # (1024, 3, 25, 3) 批次 单帧 关节点数据 三维
                    if pad_left_pose != 0 or pad_right_pose != 0:
                         # 如果需要填充，使用边缘值填充
                        self.batch_3d_in[i] = np.pad(current_segment, 
                                                ((pad_left_pose, pad_right_pose), (0, 0), (0, 0)), 
                                                'edge')
                    else:
                        # 如果不需要填充，直接赋值
                        self.batch_3d_in[i] = current_segment
                    
                    # (1024, 1, 25, 3) 批次 单帧 关节点数据 三维
                    current_segment_single_frame = current_segment[-1]
                    # 处理3D目标数据（不带填充），用于评估损失
                    if current_segment is not None:
                        self.batch_out[i] = current_segment_single_frame

                    # 处理相机参数（本实现中不使用）
                    if self.cameras is not None:
                        assert print("相机参数非空")

                # 返回数据 - 3D输入数据和目标数据
                # [:len(chunks)]确保数据长度匹配
                # 避免无效数据：防止将未初始化的缓冲区数据传递给训练代码
                # 处理边界情况：正确处理最后一批可能不足 batch_size 的数据
                if self.poses_3d is not None and self.cameras is None:
                    # 返回输入3D数据和目标数据（输入是带填充的，目标是不带填充的）
                    # 格式：(相机参数, 评估用数据, 网络输入数据)
                    yield None, self.batch_out[:len(chunks)], self.batch_3d_in[:len(chunks)]
                else:
                    print("数据生成器无数据")
                    assert False  # 如果没有数据则输出失败
            
            # 如果是无限循环模式，重置状态
            if self.endless:
                self.state = None
            else:
                # 否则退出循环
                enabled = False

class UnchunkedGenerator:
    """
    非批量数据生成器，用于测试阶段
    序列一次返回一个（即批次大小=1），不进行分块处理

    如果启用了数据增强，批次包含两个序列（即批次大小=2）,
    第二个是第一个的镜像版本

    参数说明：
    cameras -- 摄像机列表，每个视频一个元素（可选，用于半监督训练）
    poses_3d -- 3D姿态列表，每个视频一个元素（输入3D姿态数据）
    poses_valid_2d -- 2D关键点输入列表（在3D输入中不使用，应为None）
    pad -- 2D输入填充，补偿有效卷积，每边（取决于感受野）
    causal_shift -- 使用因果卷积时的不对称填充偏移（通常为0或"pad"）
    augment -- 通过水平翻转姿态进行数据增强（3D输入不使用）
    kps_left 和 kps_right -- 如果启用翻转，则为左/右2D关键点列表（3D输入不使用）
    joints_left 和 joints_right -- 如果启用翻转，则为左/右3D关节列表
    """
    def __init__(self, cameras, poses_3d, poses_valid_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        # cameras: 摄像机参数列表（可选，用于半监督训练）
        # poses_3d: 3D姿态列表（输入3D数据）
        # poses_valid_2d: 2D关键点列表（不使用）
        # pad: 3D输入填充，补偿有效卷积（每边）
        # causal_shift: 因果卷积时的不对称填充偏移量
        # augment: 是否水平翻转姿态进行数据增强（对于3D输入不使用）
        # kps_left/right: 左/右2D关键点索引（不使用）
        # joints_left/right: 左/右3D关节索引

        # 对于3D输入，禁用数据增强
        self.augment = False
        self.kps_left = kps_left  # 左侧关键点索引
        self.kps_right = kps_right  # 右侧关键点索引
        self.joints_left = joints_left  # 左侧关节索引
        self.joints_right = joints_right  # 右侧关节索引

        # 填充参数
        self.pad = pad  # 填充大小
        self.causal_shift = causal_shift  # 因果偏移量

        # 初始化数据列表，如果为None则设为空列表
        self.cameras = cameras # 摄像机参数列表置为空
        self.poses_3d = poses_3d 
        # 不使用poses_valid_2d（原2D数据）
        self.poses_3d_in = [] if poses_3d is None else poses_3d  # 3D输入数据
        # print('pose_3d',poses_3d[1].shape)
    def num_frames(self):
        """
        计算总帧数
        返回: 所有3D姿态序列的帧数总和
        """
        count = 0  # 初始化计数器
        for p in self.poses_3d:  # 遍历所有3D姿态序列
            count += p.shape[0]  # 累加每个序列的帧数
        return count

    def augment_enabled(self):
        """
        检查是否启用了数据增强
        返回: 布尔值，表示是否启用数据增强
        """
        return self.augment

    def set_augment(self, augment):
        """
        设置数据增强开关
        参数: augment - 布尔值，是否启用数据增强
        """
        self.augment = augment

    def next_epoch(self):
        """
        生成下一个epoch的数据
        遍历所有3D姿态序列，为每个序列添加填充并返回
        """
        # 遍历所有3D姿态序列及其索引
        for i, seq_3d in enumerate(self.poses_3d):
            # 扩展3D序列以包括填充
            # 为网络提供足够的时间上下文，使其能够看到感受野范围内的数据
            padded_seq_3d = np.pad(seq_3d,
                                  ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                                  'edge')  # 使用边缘值填充，保持边界特征
            
            # 扩展一个维度使其成为批次格式 (batch_size=1, ...)
            # 这样可以与训练代码兼容，即使批次大小为1
            batch_3d_in = np.expand_dims(padded_seq_3d, axis=0)  # 输入3D数据（带填充）
            batch_3d_out = np.expand_dims(seq_3d, axis=0)        # 目标3D数据（不带填充）
            
            # 如果有相机参数，也扩展维度
            batch_cam = None  # 初始化相机参数批次
            if self.cameras and i < len(self.cameras):
                # 如果存在相机参数且索引有效，则扩展维度
                batch_cam = np.expand_dims(self.cameras[i], axis=0)

            # 返回数据：(相机参数, 目标3D数据, 输入3D数据)
            # 对于3D输入，目标数据是原始3D数据，输入数据是带填充的3D数据
            yield batch_cam, batch_3d_out, batch_3d_in

# 测试代码
if __name__ == '__main__':
    from dataset.custom_dataset import linkhand_onehand_dataset
    causal_shift = 0 # 作用：表示因果偏移量
    receptive_field = 3
    pad = (receptive_field -1) // 2
    stride = 1
    batch_size = 1024
    #加载数据集
    # dataset_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\glove_data.h5'
    dataset_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove.h5'
    hand_type = 'right'
    dataset = linkhand_onehand_dataset(dataset_path, data_hand_type=hand_type)
    #数据筛选器
    action_filter = None
    keys = dataset.all_keys()
    
    # 随机取六分之五作为训练集，剩下的作为验证集
    # 获取当前时间秒
    import time
    timestampseed = int(time.time())
    np.random.seed(timestampseed)  # 设置随机种子以确保结果可复现
    print(f"随机种子: {timestampseed}")
    keys_array = np.array(keys)
    n_total = len(keys_array)
    n_train = int(n_total * 5 / 6)  # 计算训练集数量（六分之五）
    # 随机打乱索引
    shuffled_indices = np.random.permutation(n_total)
    train_indices = shuffled_indices[:n_train]  # 前n_train个作为训练集
    val_indices = shuffled_indices[n_train:]    # 剩余的作为验证集
    
    # 根据索引获取对应的键
    subjects_train = keys_array[train_indices].tolist()
    subjects_val = keys_array[val_indices].tolist()
    
    print(f"总共有 {n_total} 个subjects")
    print(f"训练集: {len(subjects_train)} 个 subjects: {subjects_train}")
    print(f"验证集: {len(subjects_val)} 个 subjects: {subjects_val}")
    
    poses_train= dataset.fetch(subjects_train, action_filter)
    #新建一个随机数组，用于测试
    # poses_train = np.random.rand(1024, 1, 17, 3)
    # train_generator = ChunkedGenerator(batch_size//stride, cameras=None, poses_3d = poses_train, poses_2d=None, chunk_length=stride,
    #                                    pad=pad, causal_shift=causal_shift, shuffle=False, augment=False,
    #                                    kps_left=None, kps_right=None, joints_left=None, joints_right=None)

    # # 获取下一个epoch的数据
    # # 循环特性分析：
    # # 循环类型：for 循环，遍历 train_generator.next_epoch() 生成的数据批次
    # # 迭代对象：ChunkedGenerator 类的 next_epoch() 方法返回的是一个生成器
    # # 循环内容：每次迭代获取一个批次的数据（batch_cam, batch_3d_out, batch_3d_in），并打印输出数据的维度
    # # 循环行为：
    # # train_generator.next_epoch() 方法内部会：
    # # 遍历所有的数据对 (pairs)
    # # 按照 batch_size 分批处理数据
    # # 每次 yield 返回一个批次的数据
    # # for 循环会持续执行直到所有批次都被处理完毕
    # # 这是一个典型的数据批处理循环，常用于深度学习训练过程中遍历数据集。

    # for batch_cam, batch_3d_out_single_frame, batch_3d_in in train_generator.next_epoch():
    #     # 打印输出的维度
    #     print("输出维度：", batch_3d_in.shape)
    #     print("单帧输出维度：", batch_3d_out_single_frame.shape)
    #     print("适应感受野多帧输出：", batch_3d_in[-1,:,1,:])
    #     print("单帧输出",batch_3d_out_single_frame[-1,:,1,:])

    test_generator = UnchunkedGenerator(cameras=None, poses_3d=poses_train, poses_valid_2d=None,
                                            pad=pad, causal_shift=causal_shift,
                                            augment=False, kps_left=None, kps_right=None,
                                            joints_left=None, joints_right=None)
    for batch_cam, batch_3d_out_single_frame, batch_3d_in in test_generator.next_epoch():
        print("测试输出维度：", batch_3d_in.shape)
        print("测试单帧输出维度：", batch_3d_out_single_frame.shape)
        

