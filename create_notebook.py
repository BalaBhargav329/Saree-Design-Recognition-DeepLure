import json

def create_notebook():
    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\n' for line in text.split('\n')]
        })
        
    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in text.split('\n')]
        })

    add_md("# AIE-CASE - Color-Invariant Saree Design Recognition\n\nThis notebook contains the complete pipeline for training and evaluating a PyTorch model to recognize saree designs independent of their color.\n\n## 1. Setup and Imports")
    add_code("""import os
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")""")

    add_md("## 2. Dataset Preparation")
    add_code("""class SareeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback if corrupt
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def split_dataset(base_dir, seed=42):
    random.seed(seed)
    
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    gallery_paths, gallery_labels = [], []
    query_paths, query_labels = [], []
    
    for c in classes:
        c_dir = os.path.join(base_dir, c)
        imgs = [os.path.join(c_dir, img) for img in os.listdir(c_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(imgs)
        
        n = len(imgs)
        if n < 4:
            train_paths.extend(imgs)
            train_labels.extend([class_to_idx[c]] * n)
            continue
            
        n_query = max(1, int(0.1 * n))
        n_gallery = max(1, int(0.1 * n))
        n_val = max(1, int(0.2 * n))
        n_train = n - n_query - n_gallery - n_val
        
        q = imgs[:n_query]
        g = imgs[n_query:n_query+n_gallery]
        v = imgs[n_query+n_gallery:n_query+n_gallery+n_val]
        t = imgs[n_query+n_gallery+n_val:]
        
        query_paths.extend(q); query_labels.extend([class_to_idx[c]] * len(q))
        gallery_paths.extend(g); gallery_labels.extend([class_to_idx[c]] * len(g))
        val_paths.extend(v); val_labels.extend([class_to_idx[c]] * len(v))
        train_paths.extend(t); train_labels.extend([class_to_idx[c]] * len(t))
        
    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'gallery': (gallery_paths, gallery_labels),
        'query': (query_paths, query_labels),
        'num_classes': len(classes)
    }""")

    add_md("## 3. Augmentations")
    add_code("""def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.8, hue=0.5)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms""")

    add_md("## 4. Model Architecture")
    add_code("""class SareeEmbeddingNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super(SareeEmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Pretrained backbone
        backbone = models.efficientnet_b0(pretrained=True)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection head to compact embedding
        self.embedding = nn.Linear(1280, embedding_dim)
        
        # Classification head for training
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        embed = self.embedding(x)
        embed_norm = F.normalize(embed, p=2, dim=1)
        
        logits = self.classifier(embed_norm)
        
        return embed_norm, logits""")

    add_md("## 5. Training Loop")
    add_code("""def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, logits = model(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} - Train Acc: {correct/total:.4f}, Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model""")

    add_md("## 6. Evaluation Functions")
    add_code("""def extract_embeddings(model, loader, device='cuda'):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            embeds, _ = model(inputs)
            all_embeddings.append(embeds.cpu())
            all_labels.extend(labels.numpy())
            
    return torch.cat(all_embeddings, dim=0), np.array(all_labels)

def evaluate_identification(query_embeds, query_labels, gallery_embeds, gallery_labels):
    sim_matrix = torch.mm(query_embeds, gallery_embeds.t())
    _, indices = torch.sort(sim_matrix, descending=True, dim=1)
    
    recalls = {1: 0, 5: 0, 10: 0}
    num_queries = query_labels.shape[0]
    
    for i in range(num_queries):
        q_label = query_labels[i]
        sorted_g_labels = gallery_labels[indices[i].numpy()]
        
        for k in recalls.keys():
            if q_label in sorted_g_labels[:k]:
                recalls[k] += 1
                
    for k in recalls.keys():
        recalls[k] = recalls[k] / num_queries
        print(f"Recall@{k}: {recalls[k]:.4f}")
        
    return recalls

def evaluate_verification(val_embeds, val_labels, test_embeds, test_labels):
    def create_pairs(embeddings, labels, num_pairs=2000):
        pairs_a, pairs_b, y_true = [], [], []
        n = len(labels)
        if n < 2: return [], [], []
        
        # positive pairs
        while len(y_true) < num_pairs // 2:
            i, j = random.sample(range(n), 2)
            if labels[i] == labels[j]:
                pairs_a.append(embeddings[i]); pairs_b.append(embeddings[j]); y_true.append(1)
        # negative pairs
        while len(y_true) < num_pairs:
            i, j = random.sample(range(n), 2)
            if labels[i] != labels[j]:
                pairs_a.append(embeddings[i]); pairs_b.append(embeddings[j]); y_true.append(0)
                
        return torch.stack(pairs_a), torch.stack(pairs_b), np.array(y_true)

    val_pa, val_pb, val_y = create_pairs(val_embeds, val_labels, num_pairs=1000)
    test_pa, test_pb, test_y = create_pairs(test_embeds, test_labels, num_pairs=2000)
    
    if len(val_y) == 0: return
    
    val_sims = F.cosine_similarity(val_pa, val_pb).numpy()
    test_sims = F.cosine_similarity(test_pa, test_pb).numpy()
    
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.linspace(0, 1, 100):
        preds = (val_sims > thresh).astype(int)
        f1 = f1_score(val_y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal threshold chosen from Validation: {best_thresh:.4f}")
    
    test_preds = (test_sims > best_thresh).astype(int)
    acc = accuracy_score(test_y, test_preds)
    prec = precision_score(test_y, test_preds, zero_division=0)
    rec = recall_score(test_y, test_preds, zero_division=0)
    f1 = f1_score(test_y, test_preds, zero_division=0)
    try: auc = roc_auc_score(test_y, test_sims)
    except: auc = float('nan')
    
    print(f"Verification - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")""")

    add_md("## 7. Execution")
    add_code("""# Define Dataset Path
# Note: For Kaggle, change this to your input dataset directory, e.g. '/kaggle/input/saree-dataset'
DATA_DIR = 'dummy_dataset'

if not os.path.exists(DATA_DIR):
    print("Dataset directory not found! Make sure to run generate_dummy_data.py or point DATA_DIR to the actual dataset.")
else:
    splits = split_dataset(DATA_DIR, seed=SEED)
    train_paths, train_labels = splits['train']
    val_paths, val_labels = splits['val']
    gallery_paths, gallery_labels = splits['gallery']
    query_paths, query_labels = splits['query']
    num_classes = splits['num_classes']
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Gallery: {len(gallery_paths)}, Query: {len(query_paths)}")
    
    train_tf, val_tf = get_transforms()
    
    train_ds = SareeDataset(train_paths, train_labels, transform=train_tf)
    val_ds = SareeDataset(val_paths, val_labels, transform=val_tf)
    gallery_ds = SareeDataset(gallery_paths, gallery_labels, transform=val_tf)
    query_ds = SareeDataset(query_paths, query_labels, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    gallery_loader = DataLoader(gallery_ds, batch_size=16, shuffle=False, num_workers=2)
    query_loader = DataLoader(query_ds, batch_size=16, shuffle=False, num_workers=2)
    
    model = SareeEmbeddingNet(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5, device=device)
    
    model.load_state_dict(torch.load('best_model.pth'))
    gallery_embeds, gallery_lbls = extract_embeddings(model, gallery_loader, device)
    query_embeds, query_lbls = extract_embeddings(model, query_loader, device)
    print("\\n--- Identification ---")
    evaluate_identification(query_embeds, query_lbls, gallery_embeds, gallery_lbls)
    
    val_embeds, val_lbls = extract_embeddings(model, val_loader, device)
    test_embeds = torch.cat([gallery_embeds, query_embeds])
    test_lbls = np.concatenate([gallery_lbls, query_lbls])
    print("\\n--- Verification ---")
    evaluate_verification(val_embeds, val_lbls, test_embeds, test_lbls)""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('Saree_Design_Recognition.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
