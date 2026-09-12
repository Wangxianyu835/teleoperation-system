"""
多假设Transformer模块
包含SHR（Self-Hypothesis Relation）和CHI（Cross-Hypothesis Interaction）块
用于处理多个假设之间的交互关系
"""
import math
import torch
import torch.nn as nn
from functools import partial
from timm.layers import DropPath
# from timm.models.layers import DropPath

class Mlp(nn.Module):
    """
    多层感知机(MLP)模块
    用于Transformer中的前馈网络(Feed-Forward Network)
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层特征维度，默认为in_features
        out_features: 输出特征维度，默认为in_features
        act_layer: 激活函数层，默认为GELU
        drop: Dropout比率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]
            
        Returns:
            输出张量 [B, N, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    多头自注意力机制(Multi-Head Self-Attention)
    实现标准的Scaled Dot-Product Attention
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数，默认为8
        qkv_bias: 是否在QKV线性层中使用偏置，默认为False
        qk_scale: 缩放因子，如果为None则使用head_dim ** -0.5
        attn_drop: 注意力Dropout比率
        proj_drop: 投影层Dropout比率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 缩放因子，用于稳定注意力计算
        self.scale = qk_scale or head_dim ** -0.5

        # 将Q、K、V合并为一个线性层以提高效率
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, N, C]，B为batch size，N为序列长度，C为特征维度
            
        Returns:
            输出张量 [B, N, C]
        """
        B, N, C = x.shape
        # 计算Q、K、V并重塑为多头形式 [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离Q、K、V

        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 应用softmax归一化
        attn = self.attn_drop(attn)

        # 应用注意力权重到V，然后重塑并投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Attention(nn.Module):
    """
    交叉注意力机制(Cross-Attention)
    用于不同假设之间的交互，Q来自一个假设，K和V来自另一个假设
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数，默认为8
        qkv_bias: 是否在线性层中使用偏置，默认为False
        qk_scale: 缩放因子，如果为None则使用head_dim ** -0.5
        attn_drop: 注意力Dropout比率
        proj_drop: 投影层Dropout比率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Q、K、V分别使用独立的线性层（与自注意力不同）
        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        """
        前向传播
        使用x_1作为Query，x_2作为Key，x_3作为Value
        
        Args:
            x_1: Query输入 [B, N, C]
            x_2: Key输入 [B, N, C]
            x_3: Value输入 [B, N, C]
            
        Returns:
            输出张量 [B, N, C]
        """
        B, N, C = x_1.shape
        # 分别计算Q、K、V
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 计算交叉注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力权重到V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SHR_Block(nn.Module):
    """
    SHR块 (Self-Hypothesis Relation Block)
    处理每个假设内部的自我关系，三个假设独立进行自注意力处理
    然后通过共享的MLP进行交互
    
    Args:
        dim: 每个假设的特征维度
        num_heads: 注意力头数
        mlp_hidden_dim: MLP隐藏层维度
        qkv_bias: 是否在QKV中使用偏置
        qk_scale: 注意力缩放因子
        drop: Dropout比率
        attn_drop: 注意力Dropout比率
        drop_path: Stochastic depth的drop path比率
        act_layer: 激活函数层
        norm_layer: 归一化层
    """
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 为三个假设分别设置归一化层
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.norm1_3 = norm_layer(dim)

        # 为三个假设分别设置自注意力层
        self.attn_1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 合并后的归一化和MLP（处理三个假设的交互）
        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2, x_3):
        """
        前向传播
        
        Args:
            x_1: 第一个假设 [B, N, dim]
            x_2: 第二个假设 [B, N, dim]
            x_3: 第三个假设 [B, N, dim]
            
        Returns:
            x_1, x_2, x_3: 更新后的三个假设 [B, N, dim]
        """
        # 每个假设独立进行自注意力处理
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm1_1(x_1)))
        x_2 = x_2 + self.drop_path(self.attn_2(self.norm1_2(x_2)))
        x_3 = x_3 + self.drop_path(self.attn_3(self.norm1_3(x_3)))

        # 合并三个假设，通过共享MLP进行交互
        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 重新分离为三个假设
        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1, x_2, x_3

class CHI_Block(nn.Module):
    """
    CHI块 (Cross-Hypothesis Interaction Block)
    处理不同假设之间的交叉交互，使用交叉注意力机制
    每个假设的更新都依赖于其他两个假设的信息
    
    Args:
        dim: 每个假设的特征维度
        num_heads: 注意力头数
        mlp_hidden_dim: MLP隐藏层维度
        qkv_bias: 是否在QKV中使用偏置
        qk_scale: 注意力缩放因子
        drop: Dropout比率
        attn_drop: 注意力Dropout比率
        drop_path: Stochastic depth的drop path比率
        act_layer: 激活函数层
        norm_layer: 归一化层
    """
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 为三个交叉注意力操作分别设置归一化层
        # 每个注意力需要三个归一化：Q、K、V各一个
        self.norm3_11 = norm_layer(dim)  # attn_1的Q (x_2)
        self.norm3_12 = norm_layer(dim)  # attn_1的K (x_3)
        self.norm3_13 = norm_layer(dim)  # attn_1的V (x_1)

        self.norm3_21 = norm_layer(dim)  # attn_2的Q (x_1)
        self.norm3_22 = norm_layer(dim)  # attn_2的K (x_3)
        self.norm3_23 = norm_layer(dim)  # attn_2的V (x_2)

        self.norm3_31 = norm_layer(dim)  # attn_3的Q (x_1)
        self.norm3_32 = norm_layer(dim)  # attn_3的K (x_2)
        self.norm3_33 = norm_layer(dim)  # attn_3的V (x_3)

        # 三个交叉注意力层，用于假设间的交互
        self.attn_1 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_2 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_3 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 合并后的归一化和MLP
        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2, x_3):
        """
        前向传播
        每个假设通过交叉注意力从其他两个假设获取信息
        
        Args:
            x_1: 第一个假设 [B, N, dim]
            x_2: 第二个假设 [B, N, dim]
            x_3: 第三个假设 [B, N, dim]
            
        Returns:
            x_1, x_2, x_3: 更新后的三个假设 [B, N, dim]
        """
        # x_1更新：Q来自x_2，K来自x_3，V来自x_1（自身）
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm3_11(x_2), self.norm3_12(x_3), self.norm3_13(x_1)))    
        # x_2更新：Q来自x_1，K来自x_3，V来自x_2（自身）
        x_2 = x_2 + self.drop_path(self.attn_2(self.norm3_21(x_1), self.norm3_22(x_3), self.norm3_23(x_2)))  
        # x_3更新：Q来自x_1，K来自x_2，V来自x_3（自身）
        x_3 = x_3 + self.drop_path(self.attn_3(self.norm3_31(x_1), self.norm3_32(x_2), self.norm3_33(x_3)))  

        # 合并三个假设，通过共享MLP进行进一步交互
        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 重新分离为三个假设
        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1, x_2, x_3

class Transformer(nn.Module):
    """
    多假设Transformer编码器
    处理三个假设（hypothesis）的交互关系
    先通过SHR块处理每个假设的内部关系，再通过CHI块处理假设间的交叉交互
    
    Args:
        depth: Transformer块的层数，默认为3
        embed_dim: 每个假设的嵌入维度，默认为512
        mlp_hidden_dim: MLP隐藏层维度，默认为1024
        h: 注意力头数，默认为8
        drop_rate: Dropout比率，默认为0.1
        length: 序列长度（位置编码的长度），默认为27
    """
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=27):
        super().__init__()
        drop_path_rate = 0.20  # Stochastic depth比率
        attn_drop_rate = 0.  # 注意力Dropout比率
        qkv_bias = True  # 启用QKV偏置
        qk_scale = None  # 注意力缩放因子

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 为三个假设分别设置位置编码
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, length, embed_dim))

        # 为三个假设分别设置位置Dropout
        self.pos_drop_1 = nn.Dropout(p=drop_rate)
        self.pos_drop_2 = nn.Dropout(p=drop_rate)
        self.pos_drop_3 = nn.Dropout(p=drop_rate)

        # Stochastic depth的线性衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        # SHR块：处理每个假设内部的自我关系
        # 使用depth-1个SHR块
        self.SHR_blocks = nn.ModuleList([
            SHR_Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-1)])

        # CHI块：处理假设间的交叉交互
        # 使用1个CHI块
        self.CHI_blocks = nn.ModuleList([
            CHI_Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth-1], norm_layer=norm_layer)
            for i in range(1)])

        # 最终归一化层（处理合并后的三个假设）
        self.norm = norm_layer(embed_dim * 3)

    def forward(self, x_1, x_2, x_3):
        """
        前向传播
        
        Args:
            x_1: 第一个假设 [B, N, embed_dim]
            x_2: 第二个假设 [B, N, embed_dim]
            x_3: 第三个假设 [B, N, embed_dim]
            
        Returns:
            输出张量 [B, N, embed_dim * 3]，三个假设合并后的特征
        """
        # 为三个假设分别添加位置编码
        x_1 += self.pos_embed_1
        x_2 += self.pos_embed_2
        x_3 += self.pos_embed_3

        # 应用位置Dropout
        x_1 = self.pos_drop_1(x_1)
        x_2 = self.pos_drop_2(x_2)
        x_3 = self.pos_drop_3(x_3)

        # 通过SHR块处理每个假设的内部关系
        for i, blk in enumerate(self.SHR_blocks):
            x_1, x_2, x_3 = self.SHR_blocks[i](x_1, x_2, x_3)

        # 通过CHI块处理假设间的交叉交互
        x_1, x_2, x_3 = self.CHI_blocks[0](x_1, x_2, x_3)

        # 合并三个假设并归一化
        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = self.norm(x)

        return x



