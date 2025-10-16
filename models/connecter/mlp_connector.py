import torch
import torch.nn as nn
from typing import Dict

class MLPConnector(nn.Module):
    """
    修改后的 MLP 连接器模块：将视觉特征映射到语言模型输入空间并输出形状为 [B, L]。
    """

    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): 包含以下关键字段：
                - input_dim: 视觉编码器输出维度，例如 768、1024、2048 等
                - output_dim: 语言模型期望的嵌入维度，例如 768（GPT2）、4096（LLaMA）
                - hidden_dim: MLP中间层大小，默认自动设为平均值
                - dropout: 可选dropout，默认0.1
        """
        super().__init__()
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        hidden_dim = config.get("hidden_dim", (input_dim + output_dim) // 2)
        dropout = config.get("dropout", 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入到第一个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(hidden_dim, hidden_dim),  # 第二个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, output_dim)  # 输出层
        )

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: [B, N, D_in] 视觉编码器输出

        Returns:
            [B,N, D_out] 映射后的特征（压缩第二个维度为1）
        """

        output_feats = self.mlp( visual_feats)  # [B, N, D_out]
        
        return output_feats
