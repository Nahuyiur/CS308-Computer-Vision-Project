# src/models/vision_encoders/transformer_encoders.py

from typing import Dict

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = '../../hf_cache'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if not os.path.exists(os.environ["HF_HOME"]):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
from transformers import AutoModel, AutoConfig

# 导入我们定义的基类
from .base_encoder import BaseVisionEncoder






class ViTEncoder(BaseVisionEncoder):
    """
    一个使用 Hugging Face `transformers` 库中预训练的 Vision Transformer (ViT)
    模型作为主干的视觉编码器。
    """

    def __init__(self, config: Dict):
        """
        初始化 ViTEncoder。

        Args:
            config (Dict): 一个包含以下键的配置字典：
                - name (str): Hugging Face Hub 上的模型名称 
                              (例如, 'google/vit-base-patch16-224-in21k')。
                - trainable (bool): 是否允许在训练过程中微调此编码器的参数。
        """
        super().__init__(config)
        
        # 1. 从 Hugging Face Hub 加载预训练模型
        # 我们使用 AutoModel 来加载不带任何特定头部的基础模型。
        model_name = self.config.get("name", "google/vit-base-patch16-224-in21k")
        self.backbone = AutoModel.from_pretrained(model_name,cache_dir = "../../hf_cache")
        self.trainable = self.config.get("trainable", False)
        # 2. 获取输出维度
        # 对于 Hugging Face 的 Transformer 模型，特征维度通常是 hidden_size。
        self._output_dim = self.backbone.config.hidden_size

        # 3. 根据配置决定是否冻结参数
        # 注意：ViT 模型对输入图像的尺寸和归一化有严格要求。
        # 这些预处理步骤应该在数据加载（Dataset）部分使用对应的 `AutoImageProcessor` 来完成。
        if not self.trainable:
            self.freeze()
        else:
            self.unfreeze()

    @property
    def output_dim(self) -> int:
        """
        返回 ViT 模型输出的特征维度 (hidden_size)。
        对于 'vit-base', 这是 768。对于 'vit-large', 这是 1024。
        """
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播，从图像中提取视觉特征。

        Args:
            images (torch.Tensor): 一个经过正确预处理（尺寸调整、归一化）的
                                   图像批次张量，形状为 (B, 3, H, W)。
                                   例如，对于标准ViT模型，H和W通常是224。

        Returns:
            torch.Tensor: ViT 输出的最后一层隐藏状态，形状为 (B, N, D)。
                          其中 N 是 patch 数量 + 1 ([CLS] token), D 是特征维度。
                          例如，对于 224x224 输入和 16x16 patch, N = 197。
        """
        # Hugging Face ViT模型期望的输入参数名是 `pixel_values`
        outputs = self.backbone(pixel_values=images)
        
        # 我们需要的是 `last_hidden_state`，它包含了每个 patch 的特征向量序列。
        # 其形状已经是 (B, N, D)，无需像 CNN 那样进行额外的维度重排。
        features = outputs.last_hidden_state
        
        return features


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 创建一个 ViT-Base 的配置
    vit_base_config = {
        # "name": "google/vit-base-patch16-224-in21k",
        "name": "google/vit-base-patch16-224",
        "trainable": False,
    }

    print(f"--- 创建 ViTEncoder (trainable=False) ---")
    # 这将从 Hugging Face Hub 下载模型权重（如果本地没有缓存）
    encoder = ViTEncoder(config=vit_base_config)
    print(f"模型已创建: {encoder.__class__.__name__}")
    print(f"模型主干: {encoder.config['name']}")
    print(f"特征输出维度: {encoder.output_dim}") # 应该为 768
    print(f"参数是否需要梯度 (冻结后): {next(encoder.parameters()).requires_grad}")

    # 2. 创建一个虚拟的输入图像批次
    # 批量大小为 4, 3个通道, 224x224 像素 (这是 ViT-Base 的标准输入尺寸)
    dummy_images = torch.randn(4, 3, 224, 224)
    
    print("\n--- 执行前向传播 ---")
    print(f"输入图像形状: {dummy_images.shape}")
    
    with torch.no_grad(): # 在推理时使用 no_grad
        output_features = encoder(dummy_images)
    
    print(f"输出特征形状: {output_features.shape}") # 应该为 torch.Size([4, 197, 768])
    # 197 = (224/16)^2 + 1 = 14*14 + 1 = 196 + 1 (CLS token)
    assert output_features.shape == (4, 197, 768)
    print("输出形状正确！✅")

    # 3. 测试解冻功能
    print("\n--- 测试解冻功能 ---")
    encoder.trainable = True # 修改配置以允许解冻
    encoder.unfreeze()
    print(f"参数是否需要梯度 (解冻后): {next(encoder.parameters()).requires_grad}")

    # 4. 检查模型是否处于训练模式
    print(f"解冻后模型是否处于训练模式: {encoder.training}")
    encoder.freeze()
    print(f"冻结后模型是否处于训练模式: {encoder.training}")