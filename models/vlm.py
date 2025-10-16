import torch
import torch.nn as nn
from models.vision_models.cnn_encoders import ResNetEncoder
from models.connecter.mlp_connector import MLPConnector
from models.connecter.qformer_connector import QFormerConnector
from transformers import AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from models.language_models.transformers_decoder import TransformerDecoder
#from models.language_models.memba_decoders import MambaDecoder

class VisionLanguageModel(nn.Module):
    def __init__(self, config):
        super(VisionLanguageModel, self).__init__()

        # 1. 初始化视觉编码器（目前只支持ResNet）
        self.vision_encoder = ResNetEncoder(config['model']['vision_encoder'])
        vision_features_dim = self.vision_encoder.output_dim
        print(f"视觉模型参数是否需要梯度 : {next(self.vision_encoder.parameters()).requires_grad}",flush =True)
        print(f"视觉模型是否处于训练模式: {self.vision_encoder.training}",flush =True)
        # 2. 初始化 Connector（仅支持MLP，目前）
        connector_type = config['model']['connector']['type'].lower()
        if connector_type == 'mlp':
            self.connector = MLPConnector(config['model']['connector']['params'])
        else:
            self.connector = QFormerConnector(config['model']['connector']['params'])

        # 3. 初始化语言模型
        decoder_type = config['model']['language_decoder']['type'].lower()
        decoder_name_or_path = config['model']['language_decoder']['name']
        
       
        self.language_decoder = TransformerDecoder(config['model']['language_decoder'])
        #else:
            #self.language_decoder = MambaDecoder(config['model']['language_decoder'])
        print(f"语言模型参数是否需要梯度 : {next(self.language_decoder.parameters()).requires_grad}",flush =True)
        print(f"语言模型是否处于训练模式: {self.language_decoder.training}",flush =True)

        self.decoder_type = decoder_type  # 用于 forward 区分处理方式

    def forward(self, images, labels, attention_mask=None):
        # 1. 提取视觉特征并转换维度
        visual_features = self.vision_encoder(images)                    # [B, N, vision_dim]
        mlp_features = self.connector(visual_features)                   # [B, N, hidden_dim]

        # 2. 文本标签部分 embedding（不包括最后一个 token）
        text_input_ids = labels[:, :-1]                                  # [B, T-1]
        text_labels = labels[:, 1:]                                      # [B, T-1] 用于计算损失（预测的是下一个词）

        text_embeds = self.language_decoder.wte(text_input_ids)          # [B, T-1, hidden_dim]

        # 3. 拼接视觉特征和文本嵌入
        inputs_embeds = torch.cat([mlp_features, text_embeds], dim=1)    # [B, N + T-1, hidden_dim]

        # 4. 构造 attention mask
        B, N = mlp_features.shape[:2]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1]                      # [B, T-1]
            visual_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=attention_mask.device)
            full_mask = torch.cat([visual_mask, attention_mask], dim=1)  # [B, N + T-1]
        else:
            full_mask = None

        # 5. 前向语言模型
        outputs = self.language_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask
        )

        logits = outputs.logits[:, N:, :]                                # [B, T-1, vocab_size]，跳过视觉部分的输出
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), text_labels.reshape(-1))  # 展平计算loss

        return loss


    def generate_sentences(self, images):
            """
            提取视觉特征并通过 MLP 连接器进行处理，返回生成的特征。
            """
            visual_features = self.vision_encoder(images)  # 从视觉编码器提取特征
            mlp_features = self.connector(visual_features)  # 通过 MLP 连接器处理特征
            text_input = "A picture of A"

# 将文本转换为 token IDs
            tokenizer = GPT2Tokenizer.from_pretrained("/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/models/pretrained_models/Qwen2.5-3B")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有 GPU，可用则选择 GPU
            text_input_ids = tokenizer.encode(text_input, return_tensors='pt')
            # 将输入和嵌入传输到 device
            text_input_ids = text_input_ids.to(device)  # 将 token IDs 移动到 device 上
            text_embeds = self.language_decoder.wte(text_input_ids).to(device)  # 获取文本嵌入并将其移动到 device 上
            mix_embeds = torch.cat([mlp_features, text_embeds], dim=1)
            with torch.no_grad():
                outputs = self.language_decoder.generate(
                    max_length=16,
                    visual_embeds = mix_embeds
                )
            return outputs