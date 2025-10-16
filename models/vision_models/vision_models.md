好的，这是一个为你的视觉编码器模块精心编写的 README 文件。

---

# 视觉编码器模块 (Vision Encoders)

欢迎使用视觉编码器模块！本模块提供了一系列可插拔的、预训练的视觉主干网络，用于从图像中提取高级特征。它是整个视觉语言模型（VLM）的“眼睛” 👀。

本模块的核心设计思想是**可扩展性**和**模块化**。通过一个统一的接口，你可以轻松地在不同的视觉架构（如 CNN, Transformer, Mamba）之间切换，以进行消融研究和性能比较。

## 核心理念：`BaseVisionEncoder`

所有编码器的基础是定义在 `base_encoder.py` 中的抽象基类 `BaseVisionEncoder`。确保了所有具体的编码器实现都遵循相同的规则，从而可以被上层模型无缝调用。

任何编码器都必须满足以下两个核心要求：

1.  **`forward(images)` 方法**: 接收一个批次的图像张量 `(B, C, H, W)`，并返回一个视觉特征序列 `(B, N, D)`。
2.  **`output_dim` 属性**: 返回一个整数，代表输出特征的维度 `D`。这个属性对于动态构建连接器至关重要。

此外，基类还提供了通用的 `.freeze()` 和 `.unfreeze()` 方法，方便在训练时控制是否微调主干网络的参数。

---

## 现有编码器

本模块目前支持以下三种主流架构：

| 编码器类 | 文件 | 架构类型 | 依赖库 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| **`ResNetEncoder`** | `cnn_encoders.py` | 卷积神经网络 (CNN) | `torchvision` | 经典且强大的CNN架构，作为坚实的基线。 |
| **`ViTEncoder`** | `transformer_encoders.py` | Vision Transformer | `transformers` | 将图像看作Patch序列，在大型数据集上表现优异。 |
| **`MambaVisionEncoder`** | `mamba_encoders.py` | 状态空间模型 (SSM) | `transformers` | 一种新兴架构，在处理长序列时具有线性的计算复杂度。 |

---

## API 与使用方法 🚀

使用本模块的最佳方式是通过配置文件来驱动。你无需在主代码中硬编码模型选择，只需修改配置文件即可。

### 1. 通过配置选择编码器

在你的 `.yaml` 配置文件中，可以这样定义想用的视觉编码器：

**示例：使用 ResNet-50**
```yaml
model:
  vision_encoder:
    name: "resnet50"
    type: "cnn" # 自定义一个类型字段，方便工厂函数构建
    params:
      pretrained: True
      trainable: False # 初始训练时冻结
```

**示例：使用 ViT-Base**
```yaml
model:
  vision_encoder:
    name: "google/vit-base-patch16-224-in21k"
    type: "transformer"
    params:
      trainable: True # 微调ViT
```

**示例：使用 Vim-Tiny**
```yaml
model:
  vision_encoder:
    name: "hustvl/vim-tiny"
    type: "mamba"
    params:
      trainable: True
```

### 2. 在代码中实例化和使用

一个工厂函数（例如 `build_vision_encoder`）会读取上述配置，并实例化正确的编码器类。以下是如何直接使用这些类的示例：

```python
import torch
from cnn_encoders import ResNetEncoder
from transformer_encoders import ViTEncoder

# --- 示例 1: 使用 ResNetEncoder ---
print("--- 演示 ResNetEncoder ---")
# 1. 定义配置
resnet_config = {
    "name": "resnet50",
    "pretrained": True,
    "trainable": False,
}

# 2. 实例化模型
resnet_encoder = ResNetEncoder(config=resnet_config)
print(f"输出维度: {resnet_encoder.output_dim}") # -> 2048

# 3. 创建伪数据并前向传播
dummy_images = torch.randn(4, 3, 224, 224)
features = resnet_encoder(dummy_images)
print(f"输出特征形状: {features.shape}") # -> torch.Size([4, 49, 2048])
print("-" * 25)


# --- 示例 2: 使用 ViTEncoder ---
print("--- 演示 ViTEncoder ---")
# 1. 定义配置
vit_config = {
    "name": "google/vit-base-patch16-224-in21k",
    "trainable": True,
}

# 2. 实例化模型
vit_encoder = ViTEncoder(config=vit_config)
print(f"输出维度: {vit_encoder.output_dim}") # -> 768

# 3. 前向传播 (注意：ViT的输入需要特定的预处理)
features = vit_encoder(dummy_images)
print(f"输出特征形状: {features.shape}") # -> torch.Size([4, 197, 768])
print("-" * 25)
```

---

## 如何添加新的编码器 🧩

想集成一个新的视觉编码器（例如 ConvNeXt）？非常简单，只需三步：

1.  **创建你的编码器类**:
    在 `src/models/vision_encoders/` 目录下创建一个新的 Python 文件（或在现有文件中添加）。让你的新类继承自 `BaseVisionEncoder`。

    ```python
    # in e.g., cnn_encoders.py
    from .base_encoder import BaseVisionEncoder

    class YourNewEncoder(BaseVisionEncoder):
        def __init__(self, config: Dict):
            super().__init__(config)
            # ... 加载你的模型 ...
            self._output_dim = 1024 # 你的模型输出维度

        @property
        def output_dim(self) -> int:
            return self._output_dim

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # ... 你的前向传播逻辑 ...
            # 确保最后返回 (B, N, D) 形状的张量
            return features
    ```

<!-- 2.  **更新工厂函数**:
    前往你的模型构建工具文件（例如 `src/utils/build_utils.py`），在 `build_vision_encoder` 函数中添加一个分支来识别你的新模型。

    ```python
    # in src/utils/build_utils.py
    def build_vision_encoder(config):
        encoder_type = config.model.vision_encoder.type
        params = config.model.vision_encoder.params
        
        if encoder_type == "cnn":
            # 如果你的新模型是CNN，可以在这里添加逻辑
            if config.model.vision_encoder.name == "your_new_encoder":
                return YourNewEncoder(config=params)
            else:
                return ResNetEncoder(config=params)
        # ... 其他分支 ...
    ``` -->

3.  **创建配置文件**:
    在 `configs/` 目录下为你的新模型创建一个新的 `.yaml` 配置文件。现在你已经可以开始训练了！