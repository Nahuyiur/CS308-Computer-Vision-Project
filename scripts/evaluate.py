import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from models.vlm import VisionLanguageModel
from datasets.coco_dataset import CocoDataset
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import random
import yaml
import sys
import torch.nn as nn
sys.path.append('/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/pycocoevalcap/bleu')
from bleu import Bleu
sys.path.append('/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/pycocoevalcap/cider')
from cider import Cider

# ============================
# 自动加载 config.yaml
# ============================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)  # 使用 yaml.safe_load 来加载 YAML 文件

# ============================
# 自动评估模型并保存生成结果
# ============================

def evaluate():
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
    model_path = "/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/saved_models/model1_resnet50_Qwen2.5-3B_mlp_epoch_1.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is eos_token for GPT2

    # 加载验证集（只使用前100个样本）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CocoDataset(
        image_dir=config['datasets']['val2017'],
        annotation_file=config['datasets']['annotations_trainval2017'],
        transform=transform,
        tokenizer_name=decoder_path,
        max_length=config['model']['language_decoder']['params']['max_length']
    )
    
    # 获取所有图片ID并选择前100个不同的图像
    unique_image_ids = []
    for idx, batch in enumerate(val_dataset):
        if len(unique_image_ids) >= 100:
            break
        image_id = batch['image_id']
        if image_id not in unique_image_ids:
            unique_image_ids.append(image_id)
    print(f"Selected image IDs: {unique_image_ids}", flush=True)
    
    # 根据这些image_ids筛选数据集
    selected_samples = [idx for idx, batch in enumerate(val_dataset) if batch['image_id'] in unique_image_ids]
    val_subset = Subset(val_dataset, selected_samples)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    # 推理并生成 caption
    results = []
    references = {}
    hypotheses = {}

    # 创建一个字典，按图片ID分组相同的图片
    image_groups = {}

    # 生成每个batch的caption并将它们按image_id分组
    for batch in tqdm(val_loader, desc="Generating captions"):
        image_id = batch['image_id'].item()  # 获取当前图片的ID
        image = batch['image'].to(device)
        input = batch['input_ids']
        outputs = model.generate_sentences(image)  # 生成视觉特征

        # 解码生成的 caption
        pred_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gt_caption = tokenizer.decode(input[0], skip_special_tokens=True)

        # 将每个image_id的caption按image_id进行分组
        if image_id not in image_groups:
            image_groups[image_id] = {'references': [], 'hypotheses': []}
        image_groups[image_id]['references'].append(gt_caption.split())  # 收集相同image_id的ground truth captions
        image_groups[image_id]['hypotheses'].append(pred_caption.split())  # 收集相同image_id的生成的caption

    # 合并相同image_id的ground truth captions
    for image_id, group in image_groups.items():
        references[image_id] = group['references']

        # 从多个生成的caption中选择最长的那个作为最终的生成结果
        longest_caption = max(group['hypotheses'], key=lambda x: len(x))  # 选择最长的那个
        hypotheses[image_id] = [longest_caption]  # 只保留最长的生成句子

        # 输出每个image的生成句子和参考句子
        generated_caption = " ".join(hypotheses[image_id][0])
        gt_caption = " ".join([caption for caption_list in references[image_id] for caption in caption_list])  # 合并所有参考caption

        results.append({
            "image_id": image_id,
            "generated_caption": generated_caption,
            "gt_caption": gt_caption,
        })

    # 使用 BLEU 和 CIDEr 计算整体的分数
    bleu = Bleu(4)  # 4为 BLEU 的最大n-gram
    cider = Cider()
    filtered_references = {}
    filtered_hypotheses = {}
    # 统计有效的 hypothesis 和 reference 数量
    count = 0
    filtered_references = {}
    filtered_hypotheses = {}

    for k, v in hypotheses.items():
        # 使用 join 将 hypothesis 拼接成一个字符串
        generated_caption = " ".join(v[0])
        
        # 检查生成的 caption 是否为空字符串
        if generated_caption.strip() != "":
            filtered_hypotheses[k] = v
            count += 1
            
            # 只有 hypothesis 非空时，才保留对应的 references
            if k in references:
                filtered_references[k] = references[k]

    print(f"有效的 hypothesis 数量: {count}")

    # 计算 BLEU 和 CIDEr 分数
    bleu_score, _ = bleu.compute_score(filtered_references, filtered_hypotheses)
    cider_score, _ = cider.compute_score(filtered_references, filtered_hypotheses)


    # 保存输出到文件
    output_dir = "eval_outputs"
    os.makedirs(output_dir, exist_ok=True)
    gen_path = os.path.join(output_dir, f"generated_captions({model_suffix}).json")

    with open(gen_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 在文件末尾添加 BLEU 和 CIDEr 分数
    metrics = {
        "BLEU-1": bleu_score[0],
        "BLEU-2": bleu_score[1],
        "BLEU-3": bleu_score[2],
        "BLEU-4": bleu_score[3],
        "CIDEr": cider_score
    }

    # 保存评分到文件
    metrics_path = os.path.join(output_dir, f"metrics({model_suffix}).json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Captions and references saved to: {gen_path}")
    print(f"✓ Metrics saved to: {metrics_path}")



'''
def evaluate():
    config_path = "/lab/haoq_lab/cse12311153/CV-Project-Image-Captioning/config.yaml"
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
    model_path = "/lab/haoq_lab/cse12311153/CV-Project-Image-Captioning/saved_models/model1_resnet50_gpt2-xl_mlp_epoch_1.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is eos_token for GPT2


    model.eval()
    results = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = CocoDataset(
        image_dir=config['datasets']['val2017'],
        annotation_file=config['datasets']['annotations_trainval2017'],
        transform=transform,
        tokenizer_name=decoder_path,
        max_length=config['model']['language_decoder']['params']['max_length']
    )
    val_subset = Subset(val_dataset, range(100))  # 只取前100个样本
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for batch in tqdm(val_loader, desc="Generating captions"):
        # 获取图像的像素值并转移到设备（GPU 或 CPU）
        pixel_values = batch["image"].to(device)
        img_ids = batch["image_id"]  # 从 batch 中提取 image_id
        
        # 生成描述
        with torch.no_grad():
            outputs = model.generate_sentences(pixel_values)  # 使用模型生成描述
        
        # 解码生成的描述
        captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # 将生成的描述和对应的 image_id 存储起来
        for img_id, caption in zip(img_ids, captions):
            results.append({
                "image_id": img_id.item(),  # 提取整数值作为 image_id
                "caption": caption
            })
    
    # 保存生成的描述到临时文件
    res_file = "temp_results.json"
    with open(res_file, 'w') as f:
        json.dump(results, f)
    
    # 使用 COCO API 进行评估
    annotation_file = "/lab/haoq_lab/cse12311153/CV-Project-Image-Captioning/datasets/annotations/captions_val2017.json"
    coco = COCO(annotation_file)  # 加载真实的 COCO 标注数据
    cocoRes = coco.loadRes(res_file)  # 加载模型生成的描述
    cocoEval = COCOEvalCap(coco, cocoRes)
    
    # 设置评估参数
    cocoEval.params['image_id'] = cocoRes.getImgIds()  # 设置评估的 image_id
    cocoEval.evaluate()  # 执行评估

    # 可选：删除临时文件以释放存储空间
    # os.remove(res_file)
    
    # 返回评估结果
    return cocoEval.eval
    '''
if __name__ == '__main__':
    #metrics = evaluate()
    #print("\nValidation Metrics:",flush=True)
    #for metric, score in metrics.items():
        #print(f"{metric}: {score:.4f}",flush=True)
    evaluate()
