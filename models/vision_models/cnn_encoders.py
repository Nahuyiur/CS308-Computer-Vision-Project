# src/models/vision_encoders/cnn_encoders.py

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
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet


# 导入我们定义的基类
from .base_encoder import BaseVisionEncoder



class ResNetEncoder(BaseVisionEncoder):
    """
    一个使用 torchvision 中预训练的 ResNet 模型作为主干的视觉编码器。

    这个类负责加载指定的 ResNet 模型，移除其用于分类的最后几层，
    并提供一个 `forward` 方法来提取空间视觉特征。
    """

    def __init__(self, config: Dict):
        """
        初始化 ResNetEncoder。

        Args:
            config (Dict): 一个包含以下键的配置字典：
                - name (str): torchvision 支持的 ResNet 模型名称 (例如, 'resnet50')。
                - pretrained (bool): 是否加载在 ImageNet 上预训练的权重。
                - trainable (bool): 是否允许在训练过程中微调此编码器的参数。
        """
        super().__init__(config)

        # 1. 加载预训练的 ResNet 模型
        model_name = self.config.get("name", "resnet50")
        pretrained = self.config.get("pretrained", True)
        weights = models.get_model_weights(model_name).DEFAULT if pretrained else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用 get_model API, 这是 torchvision 推荐的方式
        # resnet: ResNet = models.get_model(model_name, weights=weights,cache_dir=os.environ["HF_HOME"])
        resnet: ResNet = models.get_model(model_name, weights=weights)

        # 2. 获取输出维度
        # 在移除全连接层之前，我们可以方便地获取其输入维度，这就是我们想要的特征维度
        self._output_dim = resnet.fc.in_features

        # 3. 移除最后的分类层和平均池化层
        # 我们需要的是倒数第二层（最后一个卷积块）输出的空间特征图
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.backbone.to(device)

        # 4. 根据配置决定是否冻结参数
        if not self.trainable:
            self.freeze()

    @property
    def output_dim(self) -> int:
        """
        返回 ResNet 主干网络输出的特征维度。
        对于 resnet18/34, 这是 512。对于 resnet50/101/152, 这是 2048。
        """
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播，从图像中提取视觉特征。

        Args:
            images (torch.Tensor): 一个批次的图像张量，形状为 (B, 3, H, W)。

        Returns:
            torch.Tensor: 视觉特征张量，形状为 (B, N, D)，
                          其中 N 是特征图展平后的序列长度 (例如，对于224x224输入，N=49)，
                          D 是特征维度 (例如，对于ResNet50，D=2048)。
        """
        # backbone 输出的是一个空间特征图，形状为 (B, D, H', W')
        # 例如，对于 (B, 3, 224, 224) 的输入，resnet50 输出 (B, 2048, 7, 7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        features = self.backbone(images)
        
        # 将空间维度 (H', W') 展平为一个序列维度 N
        B, D, H_prime, W_prime = features.shape
        # .flatten(2) 将从第2个维度（H'）开始展平
        features = features.flatten(2)  # -> (B, D, N), 其中 N = H' * W'
        
        # 调整维度顺序以符合 (B, N, D) 的格式，这通常是 Transformer 等后续模块期望的输入格式
        features = features.permute(0, 2, 1) # -> (B, N, D)
        
        return features

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 创建一个 ResNet-50 的配置
    resnet50_config = {
        "name": "resnet50",
        "pretrained": True,
        "trainable": False, # 初始时通常先冻结
    }

    print(f"--- 创建 ResNetEncoder (trainable=False) ---")
    encoder = ResNetEncoder(config=resnet50_config)
    print(f"模型已创建: {encoder.__class__.__name__}")
    print(f"模型主干: {encoder.config['name']}")
    print(f"特征输出维度: {encoder.output_dim}") # 应该为 2048
    print(f"参数是否需要梯度 (冻结后): {next(encoder.parameters()).requires_grad}")

    # 2. 创建一个虚拟的输入图像批次
    # 批量大小为 4, 3个通道, 224x224 像素
    dummy_images = torch.randn(4, 3, 224, 224)

    # 3. 执行前向传播并检查输出形状
    output_features = encoder(dummy_images)
    print(f"输入图像形状: {dummy_images.shape}")
    print(f"输出特征形状: {output_features.shape}") # 应该为 torch.Size([4, 49, 2048])
    assert output_features.shape == (4, 49, 2048)
    print("输出形状正确！✅")

    # 4. 测试解冻功能
    print("\n--- 测试解冻功能 ---")
    encoder.config["trainable"] = True # 修改配置以允许解冻
    encoder.unfreeze()
    print(f"参数是否需要梯度 (解冻后): {next(encoder.parameters()).requires_grad}")
    
    # 5. 尝试创建一个更小的模型
    print("\n--- 创建 ResNet-18 Encoder ---")
    resnet18_config = {
        "name": "resnet18",
        "pretrained": True,
        "trainable": True,
    }
    encoder18 = ResNetEncoder(config=resnet18_config)
    print(f"模型主干: {encoder18.config['name']}")
    print(f"特征输出维度: {encoder18.output_dim}") # 应该为 512
    output18 = encoder18(dummy_images)
    print(f"ResNet-18 输出特征形状: {output18.shape}") # 应该为 torch.Size([4, 49, 512])
    assert output18.shape == (4, 49, 512)
    print("ResNet-18 输出形状正确！✅")