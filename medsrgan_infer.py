import torch
from PIL import Image
import os
from torchvision.utils import save_image
from torchvision import transforms
import argparse
from tqdm import tqdm
from medsrgan import MedSRGAN
import torch.nn.functional as F
import time

def load_model(checkpoint_path):
    """Load the trained MedSRGAN model"""
    model = MedSRGAN.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(model, image_path, output_path):
    """Process a single image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Generate SR image
    with torch.no_grad():
        sr_image = model(input_tensor)
    
    # Save output
    save_image(sr_image, output_path)
    return sr_image

def main():
    parser = argparse.ArgumentParser(description='MedSRGAN Inference')
    parser.add_argument('--checkpoint', '-p', type=str, default='best_models/medsrgan-epoch=30-val_ssim=0.9181.ckpt',
                      help='Path to model checkpoint')
    parser.add_argument('--input_dir', '-i', type=str, default='inference_data/input',
                      help='Directory containing input images')
    parser.add_argument('--output_dir', '-o', type=str, default='inference_data/output',
                      help='Directory to save super-resolved images')
    parser.add_argument('--compare', '-c', action='store_true',
                      help='Save comparison with input image')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    
    # Process all images in input directory
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images to process")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(args.input_dir, img_file)
        output_path = os.path.join(args.output_dir, f"sr_{img_file}")
        
        # Process image
        start = time.perf_counter()
        sr_image = process_image(model, input_path, output_path)
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000
        print(f"{time_ms} ms")
        
        # Create comparison if requested
        if args.compare:
            # Load original image
            lr_img = Image.open(input_path).convert('L')
            transform = transforms.Compose([transforms.ToTensor()])
            lr_tensor = transform(lr_img).unsqueeze(0)
            
            # Resize LR image to match SR size using bicubic interpolation
            lr_resized = F.interpolate(
                lr_tensor.to(device), 
                size=sr_image.shape[-2:],  # Get height and width from SR image
                mode='bicubic',
                align_corners=False
            )
            
            # Create side-by-side comparison
            comparison = torch.cat([lr_resized.cpu(), sr_image.cpu()], dim=-1)
            save_image(lr_resized, os.path.join(args.output_dir, f"lr_bicubic_{img_file}"))
            save_image(comparison, os.path.join(args.output_dir, f"compare_{img_file}"))

if __name__ == "__main__":
    main()