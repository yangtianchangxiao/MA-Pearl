"""
Graph Transformer支持工具模块
包含初始化、克隆等实用功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List
import math


def init(module: nn.Module, weight_init: Callable, bias_init: Callable, gain: float = 1.0):
    """
    通用网络层初始化函数
    
    Args:
        module: 要初始化的模块
        weight_init: 权重初始化函数
        bias_init: 偏置初始化函数  
        gain: 增益系数
    """
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    克隆网络模块
    
    Args:
        module: 要克隆的模块
        n: 克隆数量
        
    Returns:
        包含n个模块副本的ModuleList
    """
    return nn.ModuleList([type(module)(
        *[getattr(module, attr) for attr in module.__dict__ if not attr.startswith('_')]
    ) for _ in range(n)])


def create_mlp(input_dim: int, output_dim: int, hidden_dims: List[int], 
               activation: str = "relu", dropout: float = 0.0, 
               batch_norm: bool = False) -> nn.Sequential:
    """
    创建多层感知机
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度  
        hidden_dims: 隐藏层维度列表
        activation: 激活函数类型
        dropout: dropout概率
        batch_norm: 是否使用批标准化
        
    Returns:
        MLP网络
    """
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    # 激活函数映射
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU
    }
    
    act_fn = activation_map.get(activation, nn.ReLU)
    
    for i in range(len(dims) - 1):
        # 线性层
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # 最后一层不加激活函数和其他组件
        if i < len(dims) - 2:
            # 批标准化
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # 激活函数
            layers.append(act_fn())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    
    return nn.Sequential(*layers)


def orthogonal_init(layer: nn.Module, gain: float = 1.0):
    """
    正交初始化
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def xavier_uniform_init(layer: nn.Module, gain: float = 1.0):
    """
    Xavier均匀初始化
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def kaiming_normal_init(layer: nn.Module):
    """
    Kaiming正态初始化 (适合ReLU)
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_params(model: nn.Module):
    """
    冻结模型参数
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model: nn.Module):
    """
    解冻模型参数
    """
    for param in model.parameters():
        param.requires_grad = True


class PositionalEncoding(nn.Module):
    """
    位置编码，可用于Graph Transformer
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    安全的对数函数，避免log(0)
    """
    return torch.log(x.clamp(min=eps))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    带mask的softmax
    
    Args:
        logits: 输入logits
        mask: 掩码，1表示有效位置，0表示无效位置
        dim: softmax维度
    """
    # 将无效位置设为很小的值
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    return torch.softmax(masked_logits, dim=dim)


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, 
                   hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    """
    Gumbel Softmax采样
    """
    gumbels = (-torch.empty_like(logits).exponential_().log())  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits, temperature)
    y_soft = gumbels.softmax(dim)
    
    if hard:
        # 直通估计器
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    
    return ret


def polyak_update(target_params, source_params, tau: float):
    """
    Polyak平均更新目标网络
    
    Args:
        target_params: 目标网络参数
        source_params: 源网络参数
        tau: 更新系数 (0,1)
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(source_param.data, alpha=tau)


def compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """
    计算梯度范数
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) 
                                           for p in parameters]), norm_type)
    
    return total_norm.item()


def log_network_info(model: nn.Module, name: str = "Network"):
    """
    打印网络信息
    """
    param_count = count_parameters(model)
    print(f"=== {name} 信息 ===")
    print(f"参数数量: {param_count:,}")
    print(f"模型大小: {param_count * 4 / 1024 / 1024:.2f} MB")
    
    # 计算每种类型层的数量
    layer_counts = {}
    for module in model.modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    print("层类型统计:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:  # 只显示多于1个的层类型
            print(f"  {layer_type}: {count}")


# 测试代码
if __name__ == "__main__":
    print("=== Graph工具模块测试 ===")
    
    # 测试MLP创建
    mlp = create_mlp(
        input_dim=10,
        output_dim=1,
        hidden_dims=[64, 32],
        activation="relu",
        dropout=0.1,
        batch_norm=True
    )
    
    print(f"MLP结构:\n{mlp}")
    
    # 测试参数计数
    param_count = count_parameters(mlp)
    print(f"MLP参数数量: {param_count:,}")
    
    # 测试初始化
    mlp.apply(lambda m: orthogonal_init(m, gain=1.0))
    print("✅ 正交初始化完成")
    
    # 测试前向传播
    x = torch.randn(5, 10)
    y = mlp(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 测试masked softmax
    logits = torch.randn(3, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0],
                        [1, 1, 0, 0, 0], 
                        [1, 1, 1, 1, 0]], dtype=torch.float)
    
    masked_probs = masked_softmax(logits, mask)
    print(f"\nMasked softmax测试:")
    print(f"原始logits形状: {logits.shape}")
    print(f"Mask形状: {mask.shape}")
    print(f"输出概率形状: {masked_probs.shape}")
    print(f"每行概率和: {masked_probs.sum(dim=1)}")
    
    # 验证mask效果
    for i in range(3):
        valid_positions = mask[i].bool()
        invalid_positions = ~valid_positions
        print(f"行{i}: 有效位置概率 > 0: {(masked_probs[i][valid_positions] > 0).all()}")
        print(f"行{i}: 无效位置概率 ≈ 0: {torch.allclose(masked_probs[i][invalid_positions], torch.zeros_like(masked_probs[i][invalid_positions]), atol=1e-6)}")
    
    print("\n✅ Graph工具模块测试成功!")