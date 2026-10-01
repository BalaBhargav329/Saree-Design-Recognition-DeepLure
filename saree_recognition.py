import os
import time
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# 1. Dataset Prep
# ---------------------------------------------------------
class SareeDataset(Dataset):
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
        except Exception as e:
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
            # too few images, just put in train
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
    }

# ---------------------------------------------------------
# 2. Color-Invariant Augmentations
# ---------------------------------------------------------
def get_transforms():
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
    
    return train_transforms, val_transforms

# ---------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------
class SareeEmbeddingNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super(SareeEmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Pretrained backbone: ResNet-18
        backbone = models.resnet18(pretrained=True)
        
        # Using feature extractor (removing the final FC layer)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Linear projection head to compact embedding
        self.embedding = nn.Linear(512, embedding_dim)
        
        # Classification head for training
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # Feature vector
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        # Linear projection
        embed = self.embedding(x)
        
        # L2 normalization
        embed_norm = F.normalize(embed, p=2, dim=1)
        
        logits = self.classifier(embed_norm)
        
        return embed_norm, logits

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in pbar:
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
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                _, logits = model(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} - Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model

# ---------------------------------------------------------
# 5. Identification (Recall@K)
# ---------------------------------------------------------
def extract_embeddings(model, loader, device='cuda'):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Extracting Embeddings"):
            inputs = inputs.to(device)
            embeds, _ = model(inputs)
            all_embeddings.append(embeds.cpu())
            all_labels.extend(labels.numpy())
            
    return torch.cat(all_embeddings, dim=0), np.array(all_labels)

def evaluate_identification(query_embeds, query_labels, gallery_embeds, gallery_labels):
    # Cosine similarity matrix (N_query x N_gallery)
    # Since embeddings are L2 normalized, cosine similarity is just dot product
    sim_matrix = torch.mm(query_embeds, gallery_embeds.t())
    
    # Sort gallery by similarity
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

# ---------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------
def create_pairs(embeddings, labels, num_pairs=2000):
    pairs_a = []
    pairs_b = []
    y_true = []
    
    n = len(labels)
    
    # Check if there are enough elements to sample pairs
    if n < 2:
        return torch.empty(0), torch.empty(0), np.empty(0)

    # create positive pairs
    attempts = 0
    while len(y_true) < num_pairs // 2 and attempts < num_pairs * 10:
        i, j = random.sample(range(n), 2)
        if labels[i] == labels[j]:
            pairs_a.append(embeddings[i])
            pairs_b.append(embeddings[j])
            y_true.append(1)
        attempts += 1
            
    # create negative pairs
    attempts = 0
    while len(y_true) < num_pairs and attempts < num_pairs * 10:
        i, j = random.sample(range(n), 2)
        if labels[i] != labels[j]:
            pairs_a.append(embeddings[i])
            pairs_b.append(embeddings[j])
            y_true.append(0)
        attempts += 1
            
    if not pairs_a:
        return torch.empty(0), torch.empty(0), np.empty(0)
            
    return torch.stack(pairs_a), torch.stack(pairs_b), np.array(y_true)

def evaluate_verification(val_embeds, val_labels, test_embeds, test_labels):
    # Create pairs
    val_pa, val_pb, val_y = create_pairs(val_embeds, val_labels, num_pairs=1000)
    test_pa, test_pb, test_y = create_pairs(test_embeds, test_labels, num_pairs=2000)
    
    if len(val_y) == 0 or len(test_y) == 0:
        print("Not enough samples to create pairs for verification.")
        return
        
    # Calculate similarities
    val_sims = F.cosine_similarity(val_pa, val_pb).numpy()
    test_sims = F.cosine_similarity(test_pa, test_pb).numpy()
    
    # Find best threshold on validation
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.linspace(0, 1, 100):
        preds = (val_sims > thresh).astype(int)
        f1 = f1_score(val_y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal threshold chosen from Validation: {best_thresh:.4f}")
    
    # Evaluate on test (using gallery + query as proxy for test set)
    test_preds = (test_sims > best_thresh).astype(int)
    
    acc = accuracy_score(test_y, test_preds)
    prec = precision_score(test_y, test_preds, zero_division=0)
    rec = recall_score(test_y, test_preds, zero_division=0)
    f1 = f1_score(test_y, test_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(test_y, test_sims)
    except ValueError:
        auc = float('nan') # might happen if only 1 class in y_true
    
    print(f"Verification Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# ---------------------------------------------------------
# 7. Efficiency Report
# ---------------------------------------------------------
def efficiency_report(model, device):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # FLOPs
    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        flops = macs * 2
    except ImportError:
        flops = "thop not installed"
    
    # Latency
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
            
    latency_ms = np.mean(times[10:]) * 1000
    
    print("\n--- Efficiency Report ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Embedding Dimension: {model.embedding_dim}")
    print(f"Approximate Inference Latency: {latency_ms:.2f} ms")
    if isinstance(flops, (int, float)):
        print(f"FLOPs: {flops:,}")
    else:
        print(f"FLOPs: {flops}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dummy_dataset')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Splitting dataset...")
    splits = split_dataset(args.data_dir, seed=args.seed)
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
    
    # Set num_workers=0 to avoid multiprocessing issues on windows in basic script run
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = SareeEmbeddingNet(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    print("\n--- Training ---")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device)
    
    print("\n--- Identification ---")
    model.load_state_dict(torch.load('best_model.pth'))
    gallery_embeds, gallery_lbls = extract_embeddings(model, gallery_loader, device)
    query_embeds, query_lbls = extract_embeddings(model, query_loader, device)
    evaluate_identification(query_embeds, query_lbls, gallery_embeds, gallery_lbls)
    
    print("\n--- Verification ---")
    val_embeds, val_lbls = extract_embeddings(model, val_loader, device)
    # Using gallery+query as test set for verification pairs
    test_embeds = torch.cat([gallery_embeds, query_embeds])
    test_lbls = np.concatenate([gallery_lbls, query_lbls])
    evaluate_verification(val_embeds, val_lbls, test_embeds, test_lbls)
    
    efficiency_report(model, device)

if __name__ == '__main__':
    main()
