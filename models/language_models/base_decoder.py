import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional

class BaseDecoder:
    def __init__(self, pretrained_model_name: str):
        """
        基础解码器基类
        
        Args:
            pretrained_model_name: 预训练模型名称或路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        
        # 自动配置特殊token（兼容不同tokenizer）
        self._setup_special_tokens()
        
        # 确保模型和输入在相同设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _setup_special_tokens(self):
        """自动配置特殊token ID"""
        self.pad_token_id = (
            self.tokenizer.pad_token_id 
            or getattr(self.tokenizer, "eos_token_id", None))
        
        self.bos_token_id = (
            getattr(self.tokenizer, "bos_token_id", None) 
            or getattr(self.tokenizer, "cls_token_id", None))
        
        self.eos_token_id = (
            getattr(self.tokenizer, "eos_token_id", None) 
            or getattr(self.tokenizer, "sep_token_id", None))

        # 确保至少有一个终止token
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

    def preprocess_visual_features(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        预处理视觉特征（默认使用均值池化）
        
        Args:
            visual_embeds: [batch_size, num_patches, feat_dim]
            
        Returns:
            [batch_size, 1, hidden_size] 形状的嵌入
        """
        return visual_embeds.mean(dim=1, keepdim=True)

    def generate(
        self, 
        visual_embeds: torch.Tensor, 
        max_length: int = 32,
        **generation_kwargs
    ) -> list[str]:
        """
        生成文本的基类方法
        
        Args:
            visual_embeds: 视觉特征 [B, N, D]
            max_length: 生成的最大长度
            **generation_kwargs: 额外的生成参数
            
        Returns:
            生成的文本列表
        """
        # 确保输入在正确设备上
        visual_embeds = visual_embeds.to(self.device)
        
        # 预处理视觉特征
        visual_embeds = self.preprocess_visual_features(visual_embeds)  # [B, 1, H]
        
        # 准备起始token
        batch_size = visual_embeds.size(0)
        input_ids = torch.full(
            (batch_size, 1), 
            self.bos_token_id if self.bos_token_id is not None else self.eos_token_id,
            device=self.device,
            dtype=torch.long
        )

        # 生成文本
        generated_ids = self.model.generate(
            inputs_embeds=visual_embeds,
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            **generation_kwargs
        )

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

    def __call__(self, *args, **kwargs):
        """使实例可调用"""
        return self.generate(*args, **kwargs)


# === 使用示例 ===
if __name__ == "__main__":
    # 初始化解码器
    decoder = BaseDecoder(pretrained_model_name="bert-base-uncased")

    # 模拟视觉输入 (batch=2, 10个特征点，512维)
    dummy_visual = torch.randn(2, 10, 512).to(decoder.device)

    # 生成文本
    texts = decoder(
        dummy_visual,
        max_length=20,
        num_beams=3,  # 可以传入任意生成参数
        early_stopping=True
    )
    
    print("Generated Texts:")
    for i, text in enumerate(texts):
        print(f"Sample {i+1}: {text}")