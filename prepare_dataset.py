import os
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import argparse

def is_inverted_image(img_array):
    """
    Determines if the image has inverted values (white background)
    by checking if the mean pixel value is closer to white (255) than black (0)
    
    Args:
        img_array (numpy.ndarray): Input image array
        
    Returns:
        bool: True if image appears to have white background
    """
    mean_value = np.mean(img_array)
    return mean_value > 127  # If mean is closer to 255 than 0

def crop_center_patch(image, patch_size=512):
    """
    Crops a square patch of given size from the center of the image.
    
    Args:
        image (PIL.Image): Input image
        patch_size (int): Size of the square patch to crop
        
    Returns:
        PIL.Image: Cropped square patch
    """
    width, height = image.size
    left = (width - patch_size) // 2
    top = (height - patch_size) // 2
    right = left + patch_size
    bottom = top + patch_size
    return image.crop((left, top, right, bottom))

def crop_highest_intensity_half(image, patch_size=512):
    """
    Crops image to keep the half (left/right) with highest or lowest intensity sum
    depending on whether the image has a white or black background,
    then crops a square patch from the center.
    
    Args:
        image (PIL.Image): Input image
        patch_size (int): Size of the final square patch
        
    Returns:
        PIL.Image: Cropped square patch from the center of the relevant half
    """
    # Convert to numpy array for calculations
    img_array = np.array(image)
    
    # Get dimensions and ensure they're even
    height, width = img_array.shape[:2]
    width = width - (width % 2)  # Make width even
    height = height - (height % 2)  # Make height even
    mid_point = width // 2
    
    # Calculate sum of pixel values for each half
    left_sum = np.sum(img_array[:, :mid_point])
    right_sum = np.sum(img_array[:, mid_point:])
    
    # Check if image has white background
    is_inverted = is_inverted_image(img_array)
    
    # For inverted images (white background), keep the side with lower sum
    # For normal images (black background), keep the side with higher sum
    if (not is_inverted and left_sum > right_sum) or (is_inverted and left_sum < right_sum):
        half_image = image.crop((0, 0, mid_point, height))
    else:
        half_image = image.crop((mid_point, 0, width, height))
    
    # # Resize the half to maintain aspect ratio while ensuring it's large enough for patch
    # aspect_ratio = height / mid_point
    # new_width = int(patch_size * 1.5)  # Make it wider than patch_size to have room for cropping
    # new_height = int(new_width * aspect_ratio)
    
    # # Ensure new dimensions are large enough for patch
    # if new_height < patch_size:
    #     new_height = patch_size
    #     new_width = int(new_height / aspect_ratio)
    
    # half_image = half_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Crop center patch
    return crop_center_patch(half_image, patch_size)
    
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
            
            # Crop image to keep half with highest intensity
            img = crop_highest_intensity_half(img, patch_size=1024)
            
            # Resize to target size if needed
            # if img.size != (args.width//2, args.height):
            #     img = img.resize((args.width, args.height), Image.Resampling.LANCZOS)
            
            # Create low-res version
            # high_res_size = (args.width//(2*args.downscale_factor), args.height//args.downscale_factor)
            # low_res_size = (args.width//(2*args.downscale_factor*4), args.height//(args.downscale_factor*4))
            high_res_size = (img.width//(args.downscale_factor), img.height//args.downscale_factor)
            low_res_size = (img.width//(args.downscale_factor*4), img.height//(args.downscale_factor*4))
            img_high_res = img.resize(high_res_size, Image.Resampling.LANCZOS)
            img_low_res = img.resize(low_res_size, Image.Resampling.LANCZOS)
            
            # Save high-res and low-res versions
            high_res_path = os.path.join(args.output_dir, split_name, 'high_res', filename)
            low_res_path = os.path.join(args.output_dir, split_name, 'low_res', filename)
            
            img_high_res.save(high_res_path, 'PNG')
            img_low_res.save(low_res_path, 'PNG')
            
    print("\nDataset created with splits:")
    for split_name, files in splits.items():
        print(f"{split_name}: {len(files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare super-resolution dataset')
    parser.add_argument('--input_dir', type=str, default='D:/Code/dcm2png/png_data',
                        help='Input directory containing PNG images')
    parser.add_argument('--output_dir', type=str, default='mammography_sr_dataset_crop2',
                        help='Output directory for processed dataset')
    parser.add_argument('--width', type=int, default=1536,
                        help='Target width for high-resolution images')
    parser.add_argument('--height', type=int, default=2048,
                        help='Target height for high-resolution images')
    parser.add_argument('--downscale_factor', type=int, default=2,
                        help='Factor to reduce image size for low-resolution versions')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proportion of data for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Proportion of remaining data for validation')
    
    args = parser.parse_args()
    create_dataset(args)