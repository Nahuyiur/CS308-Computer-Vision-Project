# src/models/vision_encoders/base_encoder.py

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = '../../hf_cache'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if not os.path.exists(os.environ["HF_HOME"]):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict



class BaseVisionEncoder(nn.Module, ABC):
    """
    视觉编码器的抽象基类 (Abstract Base Class)。

    所有具体的视觉编码器（如 ResNetEncoder, ViTEncoder）都应该继承自这个类。
    它定义了视觉编码器必须具备的通用接口和功能，以确保它们可以被VLM主模型互换使用。
    """

    def __init__(self, config: Dict):
        """
        初始化基类。

        Args:
            config (Dict): 包含模型配置的字典，例如预训练权重路径、是否冻结等。
        """
        super().__init__()
        self.config = config
        self.trainable = self.config.get("trainable", False)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        所有子类必须实现的前向传播方法。

        这个方法接收一个批次的图像，并返回提取的视觉特征。

        Args:
            images (torch.Tensor): 一个批次的图像张量，形状通常为 (B, C, H, W)，
                                   其中 B=批量大小, C=通道数, H=高度, W=宽度。

        Returns:
            torch.Tensor: 提取的视觉特征张量。返回的形状可以是 (B, N, D)，
                          其中 N=特征序列长度, D=特征维度。
        """
        raise NotImplementedError("每个子类都必须实现 forward 方法！")

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        所有子类必须实现的属性，用于返回视觉特征的维度 (D)。

        连接器 (Connector) 需要这个信息来正确匹配语言模型的输入维度。

        Returns:
            int: 特征输出的维度。
        """
        raise NotImplementedError("每个子类都必须定义 output_dim 属性！")

    def freeze(self):
        """
        冻结模型的所有参数，使其在训练中不被更新。
        这在仅训练连接器和语言模型时非常有用。
        """
        print(f"Freezing parameters of {self.__class__.__name__}")
        for param in self.parameters():
            param.requires_grad = False
        self.eval() # 通常冻结后会切换到评估模式

    def unfreeze(self):
        """
        解冻模型的所有参数，使其在训练中可以被更新。
        这在进行端到端微调时使用。
        """
        if self.trainable:
            print(f"Unfreezing parameters of {self.__class__.__name__}")
            for param in self.parameters():
                param.requires_grad = True
            self.train() # 解冻后切换回训练模式
        else:
            print(f"'{self.__class__.__name__}' is configured not to be trainable. Keeping it frozen.")


# --- 如何使用的示例 ---
# 下面的代码块展示了如何继承和使用这个基类。
# 你可以把它放在一个单独的测试文件中，或者在这里用于快速验证。

if __name__ == '__main__':
    
    class DummyEncoder(BaseVisionEncoder):
        """一个用于演示目的的虚拟编码器。"""
        def __init__(self, config: Dict):
            super().__init__(config)
            # 假设这是一个简单的CNN，输出维度为512
            self._output_dim = 512
            # 虚拟的模型层
            self.dummy_conv = nn.Conv2d(3, self._output_dim, kernel_size=16, stride=16)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # images: (B, 3, 224, 224)
            features = self.dummy_conv(images) # -> (B, 512, 14, 14)
            
            # 将空间维度展平为一个序列
            B, D, H, W = features.shape
            features = features.view(B, D, H * W) # -> (B, 512, 196)
            
            # 调整维度顺序以匹配常见的 (B, N, D) 格式
            features = features.permute(0, 2, 1) # -> (B, 196, 512)
            return features

        @property
        def output_dim(self) -> int:
            return self._output_dim

    # 1. 创建配置和模型实例
    dummy_config = {"trainable": True}
    encoder = DummyEncoder(config=dummy_config)
    print(f"模型已创建: {encoder.__class__.__name__}")
    print(f"特征输出维度: {encoder.output_dim}")

    # 2. 创建一个虚拟的输入图像批次
    # 批量大小为4，3个通道，224x224像素
    dummy_images = torch.randn(4, 3, 224, 224)

    # 3. 执行前向传播
    output_features = encoder(dummy_images)
    print(f"输出特征形状: {output_features.shape}") # 应该输出: torch.Size([4, 196, 512])

    # 4. 测试冻结和解冻功能
    print("\n--- 测试参数冻结功能 ---")
    # 检查初始状态 (因为 trainable=True，默认应该是解冻的)
    print(f"初始参数是否需要梯度: {next(encoder.parameters()).requires_grad}")

    # 冻结
    encoder.freeze()
    print(f"冻结后参数是否需要梯度: {next(encoder.parameters()).requires_grad}")

    # 解冻
    encoder.unfreeze()
    print(f"解冻后参数是否需要梯度: {next(encoder.parameters()).requires_grad}")

    # 测试不可训练的情况
    print("\n--- 测试不可训练配置 ---")
    dummy_config_frozen = {"trainable": False}
    frozen_encoder = DummyEncoder(config=dummy_config_frozen)
    frozen_encoder.unfreeze() # 尝试解冻
    print(f"不可训练的编码器，尝试解冻后参数是否需要梯度: {next(frozen_encoder.parameters()).requires_grad}")