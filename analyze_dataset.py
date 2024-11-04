import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

class AnalysisDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.low_res_dir = os.path.join(root_dir, split, 'low_res')
        self.high_res_dir = os.path.join(root_dir, split, 'high_res')
        self.images = [f for f in os.listdir(self.low_res_dir) if f.endswith('.png')]
        
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        low_res = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')
        high_res = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')
        
        return self.transform(low_res), self.transform(high_res)

def calculate_dataset_statistics(data_dir, batch_size=32, num_workers=4):
    """Calculate mean and std of the dataset"""
    dataset = AnalysisDataset(data_dir, split='train')
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Initialize variables
    low_res_mean = 0.
    low_res_std = 0.
    high_res_mean = 0.
    high_res_std = 0.
    total_samples = 0
    
    print("Calculating mean...")
    # Calculate mean
    for low_res_batch, high_res_batch in tqdm(loader):
        batch_samples = low_res_batch.size(0)
        low_res_batch = low_res_batch.view(batch_samples, -1)
        high_res_batch = high_res_batch.view(batch_samples, -1)
        
        low_res_mean += low_res_batch.mean(1).sum().item()
        high_res_mean += high_res_batch.mean(1).sum().item()
        total_samples += batch_samples
    
    low_res_mean /= total_samples
    high_res_mean /= total_samples
    
    print("Calculating standard deviation...")
    # Calculate std
    for low_res_batch, high_res_batch in tqdm(loader):
        batch_samples = low_res_batch.size(0)
        low_res_batch = low_res_batch.view(batch_samples, -1)
        high_res_batch = high_res_batch.view(batch_samples, -1)
        
        low_res_std += ((low_res_batch - low_res_mean).pow(2)).mean(1).sum().item()
        high_res_std += ((high_res_batch - high_res_mean).pow(2)).mean(1).sum().item()
    
    low_res_std = np.sqrt(low_res_std / total_samples)
    high_res_std = np.sqrt(high_res_std / total_samples)
    
    # Calculate min and max values
    print("Calculating min/max values...")
    low_res_min = float('inf')
    low_res_max = float('-inf')
    high_res_min = float('inf')
    high_res_max = float('-inf')
    
    for low_res_batch, high_res_batch in tqdm(loader):
        low_res_min = min(low_res_min, low_res_batch.min().item())
        low_res_max = max(low_res_max, low_res_batch.max().item())
        high_res_min = min(high_res_min, high_res_batch.min().item())
        high_res_max = max(high_res_max, high_res_batch.max().item())
    
    stats = {
        'low_res': {
            'mean': low_res_mean,
            'std': low_res_std,
            'min': low_res_min,
            'max': low_res_max
        },
        'high_res': {
            'mean': high_res_mean,
            'std': high_res_std,
            'min': high_res_min,
            'max': high_res_max
        }
    }
    
    return stats

def main():
    # Update this path to your dataset directory
    data_dir = 'mammography_sr_dataset_crop'
    
    print("Analyzing dataset...")
    stats = calculate_dataset_statistics(data_dir)
    
    print("\nDataset Statistics:")
    print("\nLow Resolution Images:")
    print(f"Mean: {stats['low_res']['mean']:.4f}")
    print(f"Std: {stats['low_res']['std']:.4f}")
    print(f"Min: {stats['low_res']['min']:.4f}")
    print(f"Max: {stats['low_res']['max']:.4f}")
    
    print("\nHigh Resolution Images:")
    print(f"Mean: {stats['high_res']['mean']:.4f}")
    print(f"Std: {stats['high_res']['std']:.4f}")
    print(f"Min: {stats['high_res']['min']:.4f}")
    print(f"Max: {stats['high_res']['max']:.4f}")
    
    # Save statistics to file
    import json
    with open('dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print("\nStatistics saved to dataset_stats.json")

if __name__ == "__main__":
    main()