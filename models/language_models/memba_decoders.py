# src/models/language_decoders/mamba_decoders.py

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from .base_decoder import BaseLanguageDecoder


class MambaDecoder(BaseLanguageDecoder):
    """
    Mamba-based language decoder for VLM
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Model initialization
        self.model_name = config.get("pretrained_model_name_or_path", "state-spaces/mamba-130m")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # Mamba uses same tokenizer
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Load pretrained model
        self.model = MambaLMHeadModel.from_pretrained(
            self.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Generation config
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.max_length = self.max_length
        self.generation_config.temperature = self.temperature
        self.generation_config.top_k = self.top_k
        self.generation_config.top_p = self.top_p
        
        # Projection layer for visual features
        self.visual_proj = nn.Linear(512, self.model.config.d_model)
        
        # Freeze if needed
        if not self.trainable:
            self.freeze()

    def _init_special_tokens(self):
        """Initialize special tokens based on tokenizer"""
        self.bos_token_id = self.tokenizer.bos_token_id or 0
        self.eos_token_id = self.tokenizer.eos_token_id or 0
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds= visual_features,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        
        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states
        }

    def generate(
        self,
        visual_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        
        # Prepare initial input if not provided
        if input_ids is None:
            input_ids = torch.tensor(
                [[self.bos_token_id]],
                device=visual_features.device,
                dtype=torch.long
            )

        
        # Generate text
        outputs = self.model.generate(
            inputs_embeds=visual_features,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.eos_token_id,
            **kwargs
        )
        
        return outputs

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model.config.d_model


# Example usage
if __name__ == '__main__':
    config = {
        "pretrained_model_name_or_path": "state-spaces/mamba-130m",
        "trainable": True,
        "max_length": 128,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    }
    
    decoder = MambaDecoder(config)
    print(f"Initialized {decoder.__class__.__name__} with {decoder.model_name}")
    print(f"Vocab size: {decoder.vocab_size}")
    print(f"Hidden size: {decoder.hidden_size}")
    
    # Test forward pass
    dummy_input = torch.tensor([[decoder.bos_token_id, 100, 200, 300]])
    dummy_visual = torch.randn(1, 196, 512)  # Assuming vision encoder output is 512-dim
    
    output = decoder(dummy_input, visual_features=dummy_visual)
    print(f"Output logits shape: {output['logits'].shape}")
    
    # Test generation
    generated = decoder.generate(dummy_visual)
    print(f"Generated tokens: {generated[0]}")