# CV-Project-Image-Captioning

## Memvers

- 王子恒 12310401
- 芮煜涵 12310520
- 娄毅彬 12310513
- 方酉城 12310519

## 项目任务解析

好的，这个图像字幕生成项目很有趣！下面我为你整理了任务并制定了一个项目计划。

## 项目任务整理 

**核心目标：** 构建、训练和评估一个能够为图像生成相关且准确描述的视觉语言模型（VLM）。

**主要组成部分：**

1.  **构建和训练VLM (Build and train the VLM)**
    * **模型选择与集成：**
        * 选择一个**预训练的视觉编码器** (Vision Encoder)，例如：ResNet, ViT。
        * 选择一个**预训练的大型语言模型** (Large Language Model)，例如：LLaMA, Qwen2/Qwen2.5 (或根据资源选择GPT-2, Mamba-130M, Qwen3-0.6B等小型模型)。
        * 设计并实现一个**连接器** (Connector)，默认使用多层感知机 (MLP)，将视觉编码器的输出与语言模型的输入连接起来。
    * **数据准备：**
        * 下载并准备 **COCO 图像数据集**。
        * 下载并准备 **COCO 图像对应的字幕标注数据** (可以使用HuggingFace或提供的预处理文件)。
        * 构建数据集管道 (Dataset Pipeline) 以便高效加载和预处理图像及字幕。
    * **模型训练：**
        * 在COCO字幕数据集上训练你构建的VLM。
        * 建议使用 `transformers` 包进行构建，可以参考 `LLaVA` 的实现风格。
    * **模型评估：**
        * 在COCO测试集上评估训练好的模型。
        * 报告 **BLEU** 分数和 **Cider** 分数 (使用 `pycocoevalcap` 包进行计算)。

2.  **消融研究 (Ablation study)**
    * **视觉编码器对比：**
        * 探索不同视觉编码器架构 (例如：CNNs 如 ResNet, Transformer-based 如 ViT, Mamba-based) 对最终性能的影响。
        * 记录并报告你的发现。
    * **连接器对比：**
        * 探索不同连接器模块 (例如：将MLP替换为Q-former或其他模块) 对最终性能的影响。
        * (可选，可获额外加分) 如果你设计了新的连接器模块，请解释其设计原理。
    * **语言模型对比：**
        * 探索基于Transformer的解码器语言模型和基于Mamba的语言模型对最终性能的影响。
        * 对比它们在不同输入序列长度下的**时间消耗**。
        * 绘制**时间-序列长度曲线**。
        * 分析可能导致时间差异的原因。

**重要提示：**

* 项目重点在于**尝试和验证的过程**，而非追求最佳结果。
* 如果计算资源有限，可以从**较小的视觉模型和语言模型**开始。

**资源：**

* COCO字幕标注: HuggingFace 或提供的预处理文件。
* LLaVA风格实现参考: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
* Transformers教程: [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

---


## 项目计划结构


```
image_captioning_vlm/
├── data/                     # 存放原始和预处理后的COCO数据集 (图片和标注json)
│   ├── coco2014/
│   │   ├── train2014/        # 训练图片
│   │   ├── val2014/          # 验证图片
│   │   └── test2014/         # 测试图片 (如果使用)
│   └── annotations/
│       ├── captions_train2014.json
│       └── captions_val2014.json
│
├── src/                      # 项目核心源代码
│   ├── datasets/             # 数据集处理、加载和预处理
│   │   ├── __init__.py
│   │   ├── coco_dataset.py   # COCO数据集Dataset类实现
│   │   ├── transforms.py     # 图像和文本的转换/增强逻辑
│   │   └── collate_fn.py     # 自定义collate_fn (如果需要处理不同长度的序列)
│   │
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   ├── vision_encoders/  # 视觉编码器模块
│   │   │   ├── __init__.py
│   │   │   ├── base_encoder.py # (可选) 视觉编码器基类
│   │   │   ├── cnn_encoders.py # ResNet等CNN模型
│   │   │   └── transformer_encoders.py # ViT等Transformer模型
│   │   │   └── mamba_encoders.py   # Mamba视觉模型 (如果探索)
│   │   │
│   │   ├── language_models/  # 语言模型模块
│   │   │   ├── __init__.py
│   │   │   ├── base_decoder.py # (可选) 语言解码器基类
│   │   │   ├── transformer_decoders.py # GPT-2, LLaMA, Qwen2等
│   │   │   └── mamba_decoders.py   # Mamba语言模型
│   │   │
│   │   ├── connectors/       # 连接器模块
│   │   │   ├── __init__.py
│   │   │   ├── base_connector.py # (可选) 连接器基类
│   │   │   ├── mlp_connector.py
│   │   │   └── qformer_connector.py # (如果探索Q-Former)
│   │   │   └── custom_connector.py # (你设计的其他连接器)
│   │   │
│   │   └── vlm.py            # 核心VLM模型，组装视觉编码器、连接器和语言模型
│   │
│   ├── engine/               # 训练和评估的核心逻辑
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练循环、优化器、学习率调度等
│   │   └── evaluator.py      # 评估逻辑、生成字幕、调用pycocoevalcap
│   │
│   ├── utils/                # 通用工具函数
│   │   ├── __init__.py
│   │   ├── config_utils.py   # 加载和管理配置文件
│   │   ├── logging_utils.py  # 日志记录配置
│   │   ├── model_utils.py    # 模型加载、保存、参数统计等
│   │   └── build_utils.py    # 用于根据配置动态构建模型组件的工厂函数
│   │
│   └── tokenizers/           # (可选) 如果需要自定义或管理多种tokenizer
│       ├── __init__.py
│       └── huggingface_tokenizer.py # 封装Hugging Face tokenizer的加载
│
├── configs/                  # 存放实验配置文件 (例如 .yaml 或 .json)
│   ├── base_config.yaml      # 基础配置，可被其他配置继承
│   ├── resnet_gpt2_mlp.yaml  # 示例：ResNet + GPT-2 + MLP
│   ├── vit_qwen_qformer.yaml # 示例：ViT + Qwen + Q-Former
│   └── ablation_studies/
│       ├── vision_encoders/
│       │   ├── resnet50.yaml
│       │   └── vit_base.yaml
│       ├── connectors/
│       │   ├── mlp.yaml
│       │   └── q_former.yaml
│       └── language_models/
│           ├── gpt2.yaml
│           └── mamba_130m.yaml
│
├── scripts/                  # 可执行脚本
│   ├── train.py              # 主训练脚本
│   ├── evaluate.py           # 主评估脚本
│   ├── generate_caption.py   # (可选) 对单张图片生成字幕的脚本
│   └── benchmark_speed.py    # 用于测试不同模型/组件时间消耗的脚本
│
├── results/                  # 存放实验结果
│   ├── experiment_name_1/
│   │   ├── checkpoints/      # 保存的模型权重
│   │   ├── logs/             # 训练日志 (tensorboard, .log 文件)
│   │   └── eval_scores.json  # 评估结果 (BLEU, Cider等)
│   └── experiment_name_2/
│       └── ...
│
├── notebooks/                # (可选) Jupyter Notebooks 用于数据探索、可视化、快速原型验证
│   ├── data_exploration.ipynb
│   └── model_prototyping.ipynb
│
├── tests/                    # (可选但推荐) 单元测试和集成测试
│   ├── test_datasets.py
│   ├── test_models.py
│   └── test_engine.py
│
├── .gitignore
├── README.md                 # 项目说明文档
└── requirements.txt          # 项目依赖库
```

### 关键设计思想和可扩展性保证：

1.  **模块化 (Modularity):**
    * **`src/models/` 子目录:** 将视觉编码器、语言模型、连接器分别放在独立的子目录或文件中。每个组件都可以有自己的基类（可选，但有助于统一接口）和多个具体实现。
    * **`src/datasets/`:** 数据处理逻辑与模型分离。
    * **`src/engine/`:** 训练和评估逻辑与模型定义、数据处理分离。

2.  **配置驱动 (Configuration-Driven):**
    * **`configs/` 目录:** 使用配置文件 (如YAML) 来定义实验的各个方面，包括使用哪个视觉编码器、哪个语言模型、哪个连接器、超参数、数据路径等。
    * **`src/utils/config_utils.py`:** 用于加载和解析这些配置文件。
    * **`src/utils/build_utils.py`:** (关键) 包含工厂函数 (Factory Pattern)，这些函数根据配置文件中的名称或类型动态地创建和组装模型的不同部分 (视觉编码器、连接器、语言模型)。
        * 例如，`build_vision_encoder(config)` 可以根据 `config.model.vision_encoder.name` 来实例化 `ResNetEncoder` 或 `ViTEncoder`。

3.  **清晰的接口 (Clear Interfaces):**
    * (可选) 为视觉编码器、连接器、语言模型定义抽象基类 (ABCs) 或清晰的函数签名。这确保了当你替换一个组件时，它仍然能够与其他部分正确集成。
    * 例如，所有视觉编码器都应该有一个 `forward(images)` 方法并返回特定形状的特征。所有连接器都应该有一个 `forward(visual_features, text_tokens)` 方法。

4.  **Hugging Face Transformers 的利用:**
    * 对于预训练的视觉编码器 (如 ViT) 和语言模型 (如 GPT-2, LLaMA)，可以直接利用 `transformers` 库加载。你的封装代码可以很轻量。
    * `src/models/vision_encoders/transformer_encoders.py` 和 `src/models/language_models/transformer_decoders.py` 可以主要包含加载和适配 Hugging Face 模型的逻辑。

5.  **分离的训练和评估脚本 (`scripts/`):**
    * `train.py` 和 `evaluate.py` 接收配置文件路径作为参数。这使得运行不同的实验配置非常方便。
    * `benchmark_speed.py` 专门用于性能测试，如时间-序列长度曲线。

6.  **结果管理 (`results/`):**
    * 为每次实验（或每个配置）创建一个单独的子目录来保存模型、日志和评估分数，便于追踪和比较。

### 如何实现可扩展性：

* **添加新的视觉编码器:**
    1.  在 `src/models/vision_encoders/` 下创建一个新的Python文件 (例如 `my_new_encoder.py`) 或在现有文件中添加新类。
    2.  确保它符合预期的接口 (例如，有一个 `forward` 方法)。
    3.  更新 `src/utils/build_utils.py` 中的工厂函数，使其能够识别并实例化你的新编码器。
    4.  在 `configs/` 中创建一个新的配置文件，或修改现有配置，指定使用这个新的编码器。
* **添加新的连接器或语言模型:** 过程与添加视觉编码器类似。
* **进行消融研究:**
    1.  为每个消融设置创建或修改配置文件。
    2.  运行 `scripts/train.py` 和 `scripts/evaluate.py` 并传入相应的配置文件。
    3.  结果会自动保存在 `results/` 下的不同目录中，方便比较。

