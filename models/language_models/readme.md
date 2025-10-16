# TransformerDecoder 使用指南
## 安装依赖
```bash
pip install torch transformers
```
## 接口说明
### 1. 初始化
```python
from transformer_decoders import TransformerDecoder
config = {
"pretrained_model_name_or_path": "gpt2", # 模型名称/路径
"trainable": True, # 是否可训练
"max_length": 128, # 最大生成长度
"temperature": 0.7, # 生成温度(0.1-1.0)
"top_k": 50, # top-k采样
"top_p": 0.9 # 核心采样
}
decoder = TransformerDecoder(config)
```
### 2. 生成接口
```python
# 输入视觉特征 (shape: [batch, patches, features])
visual_features = torch.randn(1, 196, decoder.hidden_size)
# 基础生成
output_ids = decoder.generate(visual_features)
print(decoder.tokenizer.decode(output_ids[0]))
# 带初始文本的生成
input_ids = torch.tensor([[decoder.bos_token_id, 100, 200]])
output_ids = decoder.generate(visual_features, input_ids=input_ids)
```
### 3. 训练接口
```python
# 准备数据
input_ids = torch.tensor([[decoder.bos_token_id, 100, 200]]) # 输入token
labels = torch.tensor([[100, 200, decoder.eos_token_id]]) # 目标token
# 前向计算
outputs = decoder(
input_ids=input_ids,
visual_features=visual_features # shape: [1, num_patches, hidden_size]
)
# 计算损失
loss = torch.nn.CrossEntropyLoss()(
outputs["logits"].view(-1, decoder.vocab_size),
labels.view(-1)
)
loss.backward()
```
## 输入输出规范
### 输入参数
- visual_features: 视觉特征张量 [batch, num_patches, feature_dim]
- input_ids: 文本token ID [batch, seq_len]
### 输出格式
```python
{
"logits": Tensor, # 预测logits [batch, seq_len, vocab_size]
"past_key_values": Tuple, # 缓存键值对
"hidden_states": Tuple, # 各层隐藏状态
"attentions": Tuple # 注意力权重
}
```
## 注意事项
1. 视觉特征维度需与模型hidden_size一致
2. 大模型需要足够GPU显存
3. 首次运行会自动下载预训练权重