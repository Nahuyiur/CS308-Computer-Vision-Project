import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from models.vlm import VisionLanguageModel
from datasets.coco_dataset import CocoDataset
from torchvision import transforms  
from tqdm import tqdm
import sys
import os
import yaml

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

MAX_STEP=5000
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) 
    return config

def train(config):
    print("Initializing the VisionLanguageModel...", flush=True)
    model = VisionLanguageModel(config)
    model.to(device)
    print(f"Model initialized: {model.__class__.__name__}", flush=True)

    print("Setting up the dataset and dataloaders...", flush=True)
    train_dataset = CocoDataset(
        image_dir=config['datasets']['train2017'],
        annotation_file=config['datasets']['annotations_trainval2017'],
        transform=transform,
        tokenizer_name=config['model']['language_decoder']['name'],
        max_length=config['model']['language_decoder']['params']['max_length']
    )   
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Training dataset loaded. Number of batches: {len(train_loader)}", flush=True)

    learning_rate = float(config['training']['learning_rate'])
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(train_loader) * config['training']['num_epochs']
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    print("Loss function (CrossEntropyLoss) initialized.", flush=True)

    save_dir = "/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    vision_encoder_name = config['model']['vision_encoder']['name']
    language_decoder_name = os.path.basename(config['model']['language_decoder']['name']).replace("/", "_")
    connector_name = config['model']['connector']['type']

    for epoch in range(config['training']['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['training']['num_epochs']} ---", flush=True)
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for step, batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            
            loss = model(images=images,labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}", flush=True)

        model_save_path = os.path.join(
            save_dir, f"model1_{vision_encoder_name}_{language_decoder_name}_{connector_name}_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}", flush=True)

# 启动训练
print("Loading configuration file...", flush=True)
config = load_config('/lab/haoq_lab/cse12310520/CV-Project/CV-Project-Image-Captioning/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}", flush=True)
train(config)
