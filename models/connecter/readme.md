# 🔗 Connector 模块总览（视觉语言连接器）

本目录包含用于图像字幕生成任务中，**视觉编码器与语言模型之间的连接器模块**。连接器的作用是将视觉特征（如 CNN、ViT、Mamba 输出）映射为语言模型可接受的输入嵌入，解决模态之间维度不匹配和信息融合问题。

目前实现了两种连接器：

- ✅ `MLPConnector`：基础、快速、轻量
- ✅ `QFormerConnector`：引入查询 token 和 Transformer 的高级连接方式

---


---

## 1️⃣ MLPConnector（默认连接器）

**路径：** `mlp_connector.py`

### 🔹 特点

- 结构简单：`Linear → ReLU → Dropout → Linear`
- 参数少，运行快，适用于资源有限场景
- 仅做维度映射，无模态交互能力

### 🔹 初始化方式

```python
from mlp_connector import MLPConnector

config = {
    "input_dim": 1024,     # 来自视觉编码器的特征维度
    "output_dim": 768,     # 与语言模型嵌入维度一致
    "hidden_dim": 896,     # 可选，默认(input+output)/2
    "dropout": 0.1         # 可选
}


## 2 FormerConnector


###  特点
引入 可学习的 Query Tokens，用于从视觉特征中主动提取信息；

使用 HuggingFace 的 BertEncoder 作为 Transformer 结构；

支持配置 Transformer 层数、注意力头数等；

###  初始化方式

from qformer_connector import QFormerConnector

config = {
    "num_query_tokens": 32,         # 查询 token 数量
    "visual_feature_dim": 1024,     # 来自视觉编码器的特征维度
    "hidden_size": 768,             # 与语言模型嵌入维度一致
    "num_hidden_layers": 6,         # Transformer 层数（可选）
    "num_attention_heads": 8        # 注意力头数（可选）
}

connector = QFormerConnector(config)



