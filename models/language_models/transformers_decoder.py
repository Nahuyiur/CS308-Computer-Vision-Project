import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional, List, Union

class TransformerDecoder(nn.Module):
    """
    基于 HuggingFace Transformers 的因果语言模型解码器。
    支持通过 inputs_embeds 直接传入视觉特征进行文本生成。
    """

    def __init__(self, config: Dict):
        """
        初始化 Transformer 解码器。

        Args:
            config (Dict): 包含以下键的配置字典：
                - name (str): 预训练模型名称 (如 'gpt2', 'gpt2-xl')
                - pretrained (bool): 是否加载预训练权重 (默认 True)
                - trainable (bool): 是否可训练 (默认 False)
                - generation_params (Dict): 生成文本时的默认参数
        """
        super().__init__()
        self.config = config
        
        # 加载预训练模型和分词器
        self.model_name = self.config.get("name", "gpt2")
        pretrained = self.config.get("pretrained", True)
        self.trainable = self.config.get("trainable", False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        #self.wte = self.model.transformer.wte#这是给gpt2用的
        self.wte = self.model.get_input_embeddings()#这是给qwen用的
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # 设置特殊token
        self._setup_special_tokens()
        
        # 冻结参数（如果配置要求）
        if not self.trainable:
            self.freeze()
        else:
            self.unfreeze()

    def _setup_special_tokens(self) -> None:
        """确保必要的特殊token已被设置"""
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        
        # 如果tokenizer没有pad_token，添加一个（生成时需要）
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def device(self) -> torch.device:
        """返回模型所在设备"""
        return next(self.parameters()).device

    @property
    def hidden_size(self) -> int:
        """返回模型的隐藏层维度"""
        return self.model.config.hidden_size

    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.train()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播，返回语言模型的logits。

        Args:
            input_ids: 形状为 [B, L] 的token IDs
            inputs_embeds: 形状为 [B, L, D] 的自定义嵌入
            attention_mask: 形状为 [B, L] 的注意力掩码

        Returns:
            形状为 [B, L, vocab_size] 的预测logits
        """
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs

    def generate(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        visual_embeds: Optional[torch.Tensor] = None,
        max_length: int = 16,
        **generation_kwargs
    ) -> List[str]:
        """
        根据视觉特征生成文本。

        Args:
            visual_embeds: 形状为 [B, N, D] 的视觉特征
            max_length: 生成的最大长度
            generation_kwargs: 覆盖默认生成参数

        Returns:
            生成的文本列表
        """
        batch_size = visual_embeds.size(0)
        
        # 默认生成参数（可被传入的参数覆盖）
        default_kwargs = {
            "max_length": max_length,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        decoder_path = "/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/models/pretrained_models/Qwen2.5-3B"
        generation_kwargs = {**default_kwargs, **generation_kwargs}
        tokenizer = AutoTokenizer.from_pretrained(decoder_path, local_files_only=True)
        # 假设 device 是你想要将数据移动到的设备，例如 GPU 或 CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将文本 "A picture of" 转换为 input_ids，并将其移动到 device 上
        input_ids = tokenizer.encode("A picture of", return_tensors="pt").to(device)
        # 使用 tokenizer 将文本转为 input_ids
        # 生成文本
        generated_ids = self.model.generate(
            inputs_embeds=visual_embeds,
            max_new_tokens=32,
            **generation_kwargs
        )

        return generated_ids


# === 使用示例 ===
if __name__ == "__main__":
    # 配置示例
    config = {
        "name": "gpt2-xl",
        "pretrained": True,
        "trainable": False,  # 初始冻结
        "generation_params": {
            "max_length": 50,
            "temperature": 0.7,
            "top_p": 0.95
        }
    }

    print("--- 初始化 TransformerDecoder ---")
    decoder = TransformerDecoder(config)
    print(f"模型: {decoder.model_name}")
    print(f"可训练: {decoder.trainable}")
    print(f"隐藏层维度: {decoder.hidden_size}")
    print(f"设备: {decoder.device}")

    # 测试冻结/解冻
    print("\n--- 测试参数冻结 ---")
    print(f"参数是否需要梯度 (冻结状态): {next(decoder.parameters()).requires_grad}")
    decoder.unfreeze()
    print(f"参数是否需要梯度 (解冻后): {next(decoder.parameters()).requires_grad}")

    # 生成测试
    print("\n--- 测试文本生成 ---")
    dummy_visual = torch.randn(2, 10, decoder.hidden_size)  # 模拟视觉特征
    texts = decoder.generate(dummy_visual, max_length=20)
    for i, text in enumerate(texts):
        print(f"样本 {i+1}: {text}")