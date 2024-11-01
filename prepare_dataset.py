import os
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import argparse

def create_dataset(args):
    """Prepare dataset for super-resolution training"""
    
    # Create directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{args.output_dir}/{split}/high_res", exist_ok=True)
        os.makedirs(f"{args.output_dir}/{split}/low_res", exist_ok=True)

    # Get all PNG files
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png')]
    
    # Split dataset
    train_files, temp_files = train_test_split(image_files, train_size=args.train_split, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=args.val_split/(1-args.train_split), random_state=42)
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Process each split with tqdm
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split...")
        for filename in tqdm(files, desc=f"{split_name}"):
            # Read image
            img_path = os.path.join(args.input_dir, filename)
            img = Image.open(img_path)
            
            # Convert to RGB if grayscale
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size if needed
            if img.size != (args.width, args.height):
                img = img.resize((args.width, args.height), Image.Resampling.LANCZOS)
            
            # Create low-res version
            high_res_size = (args.width//args.downscale_factor, args.height//args.downscale_factor)
            low_res_size = (args.width//(args.downscale_factor*4), args.height//(args.downscale_factor*4))
            img_high_res = img.resize(high_res_size, Image.Resampling.LANCZOS)
            img_low_res = img.resize(low_res_size, Image.Resampling.LANCZOS)
            
            # Save high-res and low-res versions
            high_res_path = os.path.join(args.output_dir, split_name, 'high_res', filename)
            low_res_path = os.path.join(args.output_dir, split_name, 'low_res', filename)
            
            img.save(high_res_path, 'PNG')
            img_low_res.save(low_res_path, 'PNG')
            
    print("\nDataset created with splits:")
    for split_name, files in splits.items():
        print(f"{split_name}: {len(files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare super-resolution dataset')
    parser.add_argument('--input_dir', type=str, default='D:/Code/dcm2png/png_data',
                        help='Input directory containing PNG images')
    parser.add_argument('--output_dir', type=str, default='mammography_sr_dataset',
                        help='Output directory for processed dataset')
    parser.add_argument('--width', type=int, default=1400,
                        help='Target width for high-resolution images')
    parser.add_argument('--height', type=int, default=1760,
                        help='Target height for high-resolution images')
    parser.add_argument('--downscale_factor', type=int, default=2,
                        help='Factor to reduce image size for low-resolution versions')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proportion of data for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Proportion of remaining data for validation')
    
    args = parser.parse_args()
    create_dataset(args)