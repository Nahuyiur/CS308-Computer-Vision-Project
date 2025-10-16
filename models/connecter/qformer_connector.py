import torch
import torch.nn as nn
from typing import Dict
from transformers import BertConfig, BertModel


class QFormerConnector(nn.Module):
    """
    使用可学习查询token和Transformer Encoder的Q-Former连接器。
    """

    def __init__(self, config: Dict):
        """
        Args:
            config (Dict):
                - num_query_tokens: 查询token数量（如32）
                - hidden_size: Transformer隐藏层维度（也应与语言模型的hidden_size一致）
                - num_hidden_layers: Transformer层数（默认6）
                - num_attention_heads: 注意力头数（默认8）
        """
        super().__init__()

        self.num_query_tokens = config.get("num_query_tokens", 16)
        self.hidden_size = config["hidden_size"]

        # 可学习的查询token [num_query_tokens, hidden_size]
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, self.hidden_size))

        # Q-Former transformer encoder
        encoder_config = BertConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=config.get("num_attention_heads", 8),
            num_hidden_layers=config.get("num_hidden_layers", 6),
            intermediate_size=self.hidden_size * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            add_cross_attention=False
        )
        self.transformer = BertModel(encoder_config)  # 使用 BertModel 而不是 BertEncoder

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: [B, N, D_v] 视觉特征输入

        Returns:
            output_query: [B, Q, hidden_size] Q-Former编码后的query embeddings
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        visual_feats = visual_feats.to(device)
        B = visual_feats.size(0)
        N = visual_feats.size(1)
        D_v = visual_feats.size(2)  # 视觉特征的维度

        # 动态确定视觉特征的维度
        visual_proj = nn.Linear(D_v, self.hidden_size)
        visual_proj = visual_proj.to(device)

        # 投影视觉特征
        visual_embeds = visual_proj(visual_feats)  # [B, N, hidden_size]

        # 复制 query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, Q, hidden_size]

        # 构造 QK attention 输入
        concat_input = torch.cat([query_tokens, visual_embeds], dim=1)  # [B, Q+N, H]
        attention_mask = torch.ones(concat_input.size()[:2], dtype=torch.long, device=visual_feats.device)  # [B, Q+N]

        # Transformer Encoder
        encoder_outputs = self.transformer(
            inputs_embeds=concat_input,  # 使用 inputs_embeds 而不是 hidden_states
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # 提取 Q tokens 的输出部分
        output_query = encoder_outputs.last_hidden_state[:, :self.num_query_tokens, :]  # [B, Q, H]
        return output_query
