# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib
import os
import datetime
import logging
# 配置日志记录，把日志写入项目内的 logs 目录下，文件名包含时间戳，方便区分不同训练的日志
def setup_logging(path=None):
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建带有时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")#Y代表year，m代表month，d代表day，H代表hour，M代表minute，S代表second
    log_filename = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            # logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件:{log_filename}")
    return logger
def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    将一个torch函数包装起来，使其可以使用NumPy数组调用。输入和返回类型会被无缝转换。
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    '''
    生成一个确定性的随机数，范围在[min_value, max_value]之间，基于输入的data字符串。
    这个函数使用SHA-256哈希函数来确保相同的输入数据总是生成相同的随机数。
    通过将哈希值的前4个字节转换为整数，并将其映射到指定的范围内，确保了随机数的确定性和均匀分布。
    适用于需要在不同运行中保持一致性的场景，例如在训练过程中需要固定的随机种子。
    这个函数可以用于生成随机的参数、索引或其他需要随机性的值，同时确保在相同的输入下得到相同的输出，便于调试和复现实验结果。
    :param min_value:
    :param max_value:
    :param data:
    :return:
    '''
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def load_pretrained_weights(model, checkpoint):
    """
    将预训练权重加载到模型中。该函数会检查预训练权重的键是否与模型的状态字典匹配，并且大小是否一致。如果匹配，则将权重加载到模型中；否则，将丢弃不匹配的层。
    该函数还会处理预训练权重是否是通过 nn.DataParallel 保存的情况，如果是，则会去掉 "module." 前缀。
    该函数返回加载后的模型，并打印出匹配和丢弃的层数。
    :param model: 需要加载权重的模型
    :param checkpoint: 预训练权重的检查点，可以是包含 'state_dict' 键的字典，也可以是直接的状态字典
    :return: 加载权重后的模型
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model
