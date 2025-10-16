# src/models/vision_encoders/mamba_encoders.py

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = '../../hf_cache'
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if not os.path.exists(os.environ["HF_HOME"]):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

from typing import Dict

import torch
from transformers import AutoModel, AutoModelForImageClassification, AutoConfig

# 导入我们定义的基类
from base_encoder import BaseVisionEncoder



class MambaVisionEncoder(BaseVisionEncoder):
    """
    一个使用 Hugging Face `transformers` 库中预训练的 Vision Mamba (Vim)
    模型作为主干的视觉编码器。
    
    Vim 是将 Mamba（一种状态空间模型）架构应用于视觉任务的产物，
    它在处理序列数据时与 Transformer 有不同的效率和性能特点。
    """

    def __init__(self, config: Dict):
        """
        初始化 MambaVisionEncoder。

        Args:
            config (Dict): 一个包含以下键的配置字典：
                - name (str): Hugging Face Hub 上的模型名称 
                              (例如, 'hustvl/vim-tiny')。
                - trainable (bool): 是否允许在训练过程中微调此编码器的参数。
        """
        super().__init__(config)
        
        # 1. 从 Hugging Face Hub 加载预训练的 Vim 模型
        
        self.backbone = AutoModelForImageClassification.from_pretrained('nvidia/MambaVision-L3-512-21K',trust_remote_code=True)

        # model_name = self.config.get("name", "saurabhati/VMamba_ImageNet_83.6")
        # self.backbone = AutoModel.from_pretrained(model_name,cache_dir = "../../hf_cache")
        
        # 2. 获取输出维度
        # 和 ViT 类似，特征维度存储在模型的配置中
        self._output_dim = self.backbone.config.hidden_size

        # 3. 根据配置决定是否冻结参数
        # 注意：Vim 模型同样对输入图像的尺寸和归一化有严格要求，
        # 预处理步骤应在数据加载时使用对应的 `AutoImageProcessor` 完成。
        if not self.trainable:
            self.freeze()

    @property
    def output_dim(self) -> int:
        """
        返回 Vim 模型输出的特征维度 (hidden_size)。
        对于 'vim-tiny', 这是 192。
        """
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播，从图像中提取视觉特征。

        Args:
            images (torch.Tensor): 一个经过正确预处理（尺寸调整、归一化）的
                                   图像批次张量，形状为 (B, 3, H, W)。
                                   例如，对于标准Vim模型，H和W通常是224。

        Returns:
            torch.Tensor: Vim 输出的最后一层隐藏状态，形状为 (B, N, D)。
                          其中 N 是 patch 数量, D 是特征维度。
                          与ViT不同，Vim通常不包含 [CLS] token。
        """
        # Hugging Face Vim模型同样期望输入参数名为 `pixel_values`
        outputs = self.backbone(pixel_values=images)
        
        # 我们需要的是 `last_hidden_state`，其形状已经是 (B, N, D)
        features = outputs.last_hidden_state
        
        return features


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 创建一个 Vim-Tiny 的配置
    vim_tiny_config = {
        "name": "saurabhati/VMamba_ImageNet_83.6",
        "model_type": "mamba",  # 指定模型类型为 Mamba
        "trainable": False,
    }

    print(f"--- 创建 MambaVisionEncoder (trainable=False) ---")
    # 这将从 Hugging Face Hub 下载模型权重（如果本地没有缓存）
    encoder = MambaVisionEncoder(config=vim_tiny_config)
    print(f"模型已创建: {encoder.__class__.__name__}")
    print(f"模型主干: {encoder.config['name']}")
    print(f"特征输出维度: {encoder.output_dim}") # 应该为 192
    print(f"参数是否需要梯度 (冻结后): {next(encoder.parameters()).requires_grad}")

    # 2. 创建一个虚拟的输入图像批次
    # 批量大小为 4, 3个通道, 224x224 像素 (Vim 的标准输入尺寸)
    dummy_images = torch.randn(4, 3, 224, 224)
    
    print("\n--- 执行前向传播 ---")
    print(f"输入图像形状: {dummy_images.shape}")
    
    with torch.no_grad():
        output_features = encoder(dummy_images)
    
    print(f"输出特征形状: {output_features.shape}") # 应该为 torch.Size([4, 196, 192])
    # 196 = (224/16)^2 = 14*14。注意这里没有 +1 的 [CLS] token。
    assert output_features.shape == (4, 196, 192)
    print("输出形状正确！✅")

    # 3. 测试解冻功能
    print("\n--- 测试解冻功能 ---")
    encoder.config["trainable"] = True # 修改配置以允许解冻
    encoder.unfreeze()
    print(f"参数是否需要梯度 (解冻后): {next(encoder.parameters()).requires_grad}")