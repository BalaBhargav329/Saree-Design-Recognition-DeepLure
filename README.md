# AIE-CASE – Color-Invariant Saree Design Recognition

## Overview
This repository contains the complete solution for the AI/ML assignment "AIE-CASE – Color-Invariant Saree Design Recognition".
The goal is to build a robust model capable of identifying saree surface designs and patterns independently of their color palette, using PyTorch.

## Files included
1. `saree_recognition.py` - The main Python script with the entire training, identification, and verification pipeline.
2. `Saree_Design_Recognition.ipynb` - Kaggle-ready Jupyter notebook containing the same pipeline for interactive execution.
3. `Approach_Note.md` - A 500-character summary of the architectural and methodological choices.
4. `generate_dummy_data.py` - Script to generate synthetic saree images for testing the pipeline when the real dataset is not immediately available.
5. `README.md` - This file.

## Requirements
Ensure you have Python 3.8+ installed. Install the dependencies using:
```bash
pip install torch torchvision numpy pillow scikit-learn tqdm thop
```
*Note: `thop` is optional but highly recommended to calculate FLOPs for the efficiency report.*

## Dataset Structure
The code assumes the dataset is provided in a standard PyTorch `ImageFolder` format where each subfolder corresponds to a unique saree design class:
```
dataset_dir/
  design_001/
    img1.jpg
    img2.jpg
    ...
  design_002/
    img1.jpg
    ...
```
The script will automatically detect the classes and split them securely into `train`, `val`, `gallery`, and `query` splits avoiding any data leakage.

## How to Run
### 1. Generating Dummy Data (If real dataset is absent)
Run the data generator to create synthetic test data:
```bash
python generate_dummy_data.py
```
This creates a folder `dummy_dataset` with 10 classes and 20 variations per class, with completely varied colorations but identical underlying structural geometric patterns.

### 2. Running the Pipeline
To run the full pipeline (training, identification evaluation, verification evaluation, and efficiency report):
```bash
python saree_recognition.py --data_dir dummy_dataset --epochs 5 --batch_size 16 --seed 42
```
Arguments:
- `--data_dir`: Path to the root dataset folder. (Default: `dummy_dataset`)
- `--epochs`: Number of epochs for training. (Default: 5)
- `--batch_size`: Batch size for DataLoader. (Default: 16)
- `--seed`: Random seed for reproducibility. (Default: 42)

## Outputs
- **Identification**: Outputs `Recall@1`, `Recall@5`, and `Recall@10` by matching query images against the gallery.
- **Verification**: Outputs ROC-AUC, Accuracy, Precision, Recall, and F1-score for paired tests (same design vs different design). It automatically computes the best cosine similarity threshold on the validation set.
- **Efficiency Report**: Displays Total parameters, Trainable parameters, Embedding dimension, FLOPs, and Approximate inference latency.

## Architecture & Methodologies
- **Backbone**: `EfficientNet-B0` (Pre-trained) - Provides an excellent trade-off between accuracy and computational efficiency.
- **Color Invariance**: Enforced via severe color-based data augmentations: `RandomGrayscale` and aggressive `ColorJitter` (Hue, Saturation, Brightness, Contrast). This forces the CNN to rely entirely on texture, shape, and structural frequency components.
- **Embedding Space**: Features are projected into a compact dimension (128) and $L_2$-normalized. 
- **Loss**: Classification via Cross-Entropy on normalized embeddings maps directly to optimizing cosine similarity, effectively acting as a proxy metric-learning technique (simpler and often more stable than Triplet loss for this structure).
- **Matching Metric**: Cosine similarity is used for both Identification ranking and Verification thresholding.
