import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vlm import VisionLanguageModel
from datasets.coco_dataset import CocoDataset
import yaml

# ============================
# 自动加载 config.yaml
# ============================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)  # 使用 yaml.safe_load 来加载 YAML 文件

# ============================
# 生成测试集的 caption
# ============================
def generate_test_captions():
    config_path = "/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/config.yaml"
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取模型名称作为文件名后缀
    decoder_path = config['model']['language_decoder']['name']
    model_suffix = os.path.basename(decoder_path).replace('/', '_')

    # 初始化模型
    model = VisionLanguageModel(config)
    model.to(device)

    # 自动加载最近的权重文件
    save_dir = "saved_models"
    candidates = [f for f in os.listdir(save_dir) if model_suffix in f and f.endswith(".pth")]
    if not candidates:
        raise FileNotFoundError(f"No saved model with suffix {model_suffix} found in {save_dir}")
    latest_model = sorted(candidates)[-1]
    model_path = os.path.join(save_dir, latest_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is eos_token for GPT2

    # 加载测试集（前100条）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CocoDataset(
        image_dir=config['datasets']['test2017'],  # Change to test dataset if available
        annotation_file=None,  # No annotations needed for testing captions
        transform=transform,
        tokenizer_name=decoder_path,
        max_length=config['model']['language_decoder']['params']['max_length']
    )

    # 只使用前100张图片
    test_subset = torch.utils.data.Subset(test_dataset, range(100))  # 取前100个样本
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    # 推理并生成 caption
    generated_captions = []

    for batch in tqdm(test_loader, desc="Generating captions"):
        if batch is None:
            print("Skipping invalid batch.")
            continue  # 跳过无效数据

        image = batch['image'].to(device)
        input_ids = batch['input_ids']

        with torch.no_grad():
            # 推理生成 caption
            outputs = model.language_decoder.generate(
                input_ids=input_ids.to(device),
                max_new_tokens=50,
                num_beams=5,  # Beam search
                top_p=0.9,    # Nucleus sampling
                temperature=0.7,
                early_stopping=True,
                attention_mask=batch['attention_mask'].to(device)
            )

        # 解码生成的 caption
        pred_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 确保图像路径存在并保存生成的 caption
        image_path = batch['image_path']
        if not image_path:
            print("No image path found.")
            continue  # 跳过没有路径的样本

        generated_captions.append({
            "image_path": image_path,  # 获取图像路径
            "generated_caption": pred_caption
        })

    # 保存输出到文件
    output_dir = "eval_outputs"
    os.makedirs(output_dir, exist_ok=True)

    gen_path = os.path.join(output_dir, f"generated_captions({model_suffix})(test).json")

    with open(gen_path, 'w') as f:
        json.dump(generated_captions, f, indent=2)

    print(f"✓ Captions saved to: {gen_path}")

if __name__ == '__main__':
    generate_test_captions()
