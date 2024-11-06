iimport pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import time
import monai.losses

class SRDataset(Dataset):
    """
    Dataset class for Super Resolution (SR) medical images.
    Handles loading and preprocessing of paired low-resolution and high-resolution images.
    """
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: Root directory containing train/val splits
            split: Dataset split ('train' or 'val')
        """
        self.low_res_dir = os.path.join(root_dir, split, 'low_res')
        self.high_res_dir = os.path.join(root_dir, split, 'high_res')
        self.images = [f for f in os.listdir(self.low_res_dir) if f.endswith('.png')]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def normalize_minmax(self, tensor):
        """
        Normalize tensor values to [0,1] range using min-max normalization.
        Prevents division by zero for constant-valued tensors.
        """
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            return (tensor - min_val) / (max_val - min_val)
        return tensor

    def __getitem__(self, idx):
        """
        Load and preprocess image pair at given index.
        Returns normalized low-res and high-res image tensors.
        """
        img_name = self.images[idx]
        
        # Load grayscale images
        low_res_orig = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')
        high_res_orig = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')
        
        # Convert to tensors and normalize
        low_res = self.transform(low_res_orig)
        high_res = self.transform(high_res_orig)
        
        low_res = self.normalize_minmax(low_res)
        high_res = self.normalize_minmax(high_res)

        return low_res, high_res

    def apply_gaussian_noise(self, low_res, mean=0.0, std=0.1):
        """Add Gaussian noise to low-resolution images for data augmentation"""
        noise = torch.randn_like(low_res) * std + mean
        low_res = low_res + noise
        return low_res

class D_Block(nn.Module):
    """
    Basic discriminator block consisting of Conv2d + BatchNorm + LeakyReLU.
    Used as a building block in the discriminator network.
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class RWMAB(nn.Module):
    """
    Residual Weighted Multi-Attention Block (RWMAB).
    Combines residual learning with attention mechanism for better feature extraction.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Feature extraction path
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
        )
        # Attention path
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ = self.layer1(x)
        x__ = self.layer2(x_)
        # Combine attention with residual connection
        x = x__ * x_ + x
        return x

class ShortResidualBlock(nn.Module):
    """
    Stack of RWMAB blocks with a global residual connection.
    Enhances feature extraction while maintaining gradient flow.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.ModuleList([RWMAB(in_channels) for _ in range(16)])

    def forward(self, x):
        x_ = x.clone()
        for layer in self.layers:
            x_ = layer(x_)
        return x_ + x

class Generator(nn.Module):
    """
    Generator network for super-resolution.
    Architecture: Initial conv -> Residual blocks -> Upsampling layers -> Output conv
    """
    def __init__(self, in_channels=1, blocks=8):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1)
        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(64) for _ in range(blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)
        
        # Upsampling layers for 4x resolution increase
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),  # 2x upscale
            nn.Conv2d(64, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),  # 2x upscale
            nn.Conv2d(64, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x_ = x.clone()

        # Process through residual blocks
        for layer in self.short_blocks:
            x_ = layer(x_)

        # Combine processed features with input features
        x = torch.cat([self.conv2(x_), x], dim=1)
        x = self.conv3(x)
        return x

class Discriminator(nn.Module):
    """
    Dual-path discriminator network with feature extraction capabilities.
    Uses two processing paths that are later combined for robust discrimination.
    """
    def __init__(self, in_channels=1):
        super().__init__()

        # Path 1 - Rapid downsampling
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1), 
            nn.LeakyReLU()
        )
        self.block_1_1 = D_Block(64, 64, stride=2)
        self.block_1_2 = D_Block(64, 128, stride=2)
        self.block_1_3 = D_Block(128, 128, stride=1)

        # Path 2 - Gradual downsampling
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1), 
            nn.LeakyReLU()
        )
        self.block_2_1 = D_Block(64, 64, stride=2)
        self.block_2_2 = D_Block(64, 128, stride=2)

        # Combined path processing
        self.block3 = D_Block(256, 256, stride=1)
        self.block4 = D_Block(256, 256, stride=2)
        self.block5 = D_Block(256, 512, stride=1)
        self.block6 = D_Block(512, 512, stride=2)
        self.block7 = D_Block(512, 1024, stride=2)
        self.block8 = D_Block(1024, 1024, stride=2)

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def get_features(self, x):
        """Extract intermediate features for perceptual loss computation"""
        features = []
        
        # Extract features from both paths
        x_1 = self.conv_1_1(x)
        features.append(x_1)
        
        x_1 = self.block_1_1(x_1)
        features.append(x_1)
        
        x_1 = self.block_1_2(x_1)
        x_1 = self.block_1_3(x_1)
        features.append(x_1)
        
        x_2 = self.conv_2_1(x)
        x_2 = self.block_2_1(x_2)
        x_2 = self.block_2_2(x_2)
        features.append(x_2)
        
        x = torch.cat([x_1, x_2], dim=1)
        x = self.block3(x)
        features.append(x)
        
        return features
    
    def forward(self, x, get_features=False):
        if get_features:
            return self.get_features(x)
        
        # Process through both paths
        x_1 = self.conv_1_1(x)
        x_1 = self.block_1_1(x_1)
        x_1 = self.block_1_2(x_1)
        x_1 = self.block_1_3(x_1)
        
        x_2 = self.conv_2_1(x)
        x_2 = self.block_2_1(x_2)
        x_2 = self.block_2_2(x_2)

        # Combine paths and process through remaining layers
        x = torch.cat([x_1, x_2], dim=1)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        # Classification
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        return self.sigmoid(x)

class MedSRGAN(pl.LightningModule):
    """
    PyTorch Lightning module implementing SRGAN for medical image super-resolution.
    Combines adversarial training with perceptual and content losses.
    """
    def __init__(self, lr: float = 2e-4):
        super().__init__()
        self.outputs = []
        self.automatic_optimization = False
        
        # Initialize networks
        self.generator = Generator(in_channels=1)
        self.discriminator = Discriminator(in_channels=1)
        self.lr = lr
        
        # Initialize network weights
        self.generator.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)
        
        # Loss functions
        self.content_l1_loss = nn.L1Loss()
        self.content_vgg_loss = monai.losses.PerceptualLoss(
            spatial_dims=2,
            network_type="vgg",
            is_fake_3d=False,
            pretrained=True
        )
        
        # Loss weights for balanced training
        self.content_l1_weight = 0.01
        self.content_vgg_weight = 1
        self.content_weight = 1.0
        self.adversarial_weight = 0.05
        self.feature_weights = [1/2, 1/4, 1/8, 1/16, 1/16]
        self.adversarial_feature_weight = 0.005
        
        # Best metric tracking
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0
        
    def _init_weights(self, m):
        """Initialize network weights using the DCGAN strategy"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def compute_discriminator_loss(self, hr_imgs, sr_imgs):
        """
        Compute discriminator loss with label smoothing for stable training.
        Uses BCE loss between real/fake predictions and smoothed labels.
        """
        hr_logits = self.discriminator(hr_imgs)
        sr_logits = self.discriminator(sr_imgs.detach())
        
        # Label smoothing
        real_labels = torch.rand_like(hr_logits) * 0.1 + 0.9
        fake_labels = torch.rand_like(sr_logits) * 0.1
        
        d_loss_real = F.binary_cross_entropy_with_logits(hr_logits, real_labels)
        d_loss_fake = F.binary_cross_entropy_with_logits(sr_logits, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
        # Log discriminator accuracy
        with torch.no_grad():
            d_real_acc = ((torch.sigmoid(hr_logits) > 0.5).float().mean())
            d_fake_acc = ((torch.sigmoid(sr_logits) < 0.5).float().mean())
            self.log('d_real_acc', d_real_acc)
            self.log('d_fake_acc', d_fake_acc)
        
        return d_loss

    def compute_adversarial_feature_loss(self, lr_imgs, hr_imgs, sr_imgs):
        """
        Compute feature-based adversarial loss using discriminator's intermediate features.
        Encourages SR images to have similar feature distributions as HR images.
        """
        hr_features = self.discriminator.get_features(hr_imgs)
        lr_features = self.discriminator.get_features(lr_imgs)
        sr_features = self.discriminator.get_features(sr_imgs)
        
        feature_loss = 0.0
        # Compute weighted MSE for each feature layer
        for i, (w, hr_feat, lr_feat, sr_feat) in enumerate(zip(
            self.feature_weights, hr_features, lr_features, sr_features)):
            # Upsample LR features to match spatial dimensions
            lr_feat_upscaled = F.interpolate(
                lr_feat,
                size=hr_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Compute MSE between feature maps
            mse_lr_hr = F.mse_loss(lr_feat_upscaled, hr_feat)
            mse_lr_sr = F.mse_loss(lr_feat_upscaled, sr_feat)
            feature_loss += w * (mse_lr_hr + mse_lr_sr)
            
            # Log individual layer losses
            self.log(f'feature_loss_layer_{i}', w * (mse_lr_hr + mse_lr_sr))
            
        return feature_loss
    
    def compute_generator_loss(self, lr_imgs, sr_imgs, hr_imgs):
        """
        Compute total generator loss combining content, adversarial, and feature losses.
        Returns individual loss components for logging.
        """
        # Content loss combining L1 and VGG perceptual losses
        content_loss_l1 = self.content_l1_loss(sr_imgs, hr_imgs)
        content_loss_vgg = self.content_vgg_loss(sr_imgs, hr_imgs)
        content_loss = self.content_l1_weight * content_loss_l1 + self.content_vgg_weight * content_loss_vgg
                
        # Standard adversarial loss
        fake_logits = self.discriminator(sr_imgs)
        adversarial_loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits)
        )

        # Feature-based adversarial loss
        adversarial_feature_loss = self.compute_adversarial_feature_loss(lr_imgs, hr_imgs, sr_imgs)

        # Weighted sum of all losses
        total_loss = (
            self.content_weight * content_loss +
            self.adversarial_feature_weight * adversarial_feature_loss +
            self.adversarial_weight * adversarial_loss
        )
        
        return total_loss, content_loss, adversarial_loss, adversarial_feature_loss

    def training_step(self, batch, batch_idx):
        """
        Execute one training step with discriminator and generator updates.
        Uses gradient clipping and mixed precision training for stability.
        """
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
        
        # Train discriminator
        opt_d.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            sr_imgs = self(lr_imgs)
            d_loss = self.compute_discriminator_loss(hr_imgs, sr_imgs)
        
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        opt_d.step()
        
        # Train generator
        opt_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            sr_imgs = self(lr_imgs)
            g_loss, content_loss, adv_loss, percep_loss = self.compute_generator_loss(lr_imgs, sr_imgs, hr_imgs)
        
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        opt_g.step()
        
        # Log training metrics
        self.log_dict({
            'g_loss': g_loss,
            'd_loss': d_loss,
            'content_loss': content_loss,
            'adversarial_loss': adv_loss,
            'perceptual_loss': percep_loss
        }, prog_bar=True)

    def calculate_ssim(self, img1, img2):
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        Uses 11x11 Gaussian window and standard constants.
        """
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def validation_step(self, batch, batch_idx):
        """
        Perform validation step and compute quality metrics (PSNR, SSIM).
        Periodically saves validation image samples.
        """
        lr_imgs, hr_imgs = batch
        
        with torch.cuda.amp.autocast():
            sr_imgs = self(lr_imgs)
            g_loss, content_loss, adv_loss, percep_loss = self.compute_generator_loss(lr_imgs, sr_imgs, hr_imgs)
            d_loss = self.compute_discriminator_loss(hr_imgs, sr_imgs)
        
        # Calculate image quality metrics
        with torch.no_grad():
            mse = F.mse_loss(sr_imgs, hr_imgs)
            psnr = 10 * torch.log10(1.0 / mse)
            ssim_value = self.calculate_ssim(sr_imgs, hr_imgs)
            
        # Log validation metrics
        self.log_dict({
            'val/generator_loss': g_loss,
            'val/discriminator_loss': d_loss,
            'val/content_loss': content_loss,
            'val/adversarial_loss': adv_loss,
            'val/perceptual_loss': percep_loss,
            'val/psnr': psnr,
            'val/ssim': ssim_value
        }, prog_bar=True, sync_dist=True)
        
        # Save validation images every 10 epochs
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            os.makedirs('validation_images', exist_ok=True)
            
            comparison = torch.cat([
                F.interpolate(lr_imgs, size=hr_imgs.shape[-2:], mode='nearest'),
                sr_imgs,
                hr_imgs
            ], dim=0)
            
            save_image(
                comparison,
                f'validation_images/epoch_{self.current_epoch+1}.png',
                nrow=len(lr_imgs),
                normalize=True
            )
        
        return {
            'generator_loss': g_loss,
            'psnr': psnr,
            'ssim': ssim_value
        }
    
    def on_validation_epoch_end(self):
        """
        Compute and log epoch-level validation metrics.
        Updates best model tracking based on SSIM score.
        """
        avg_gen_loss = torch.stack([x['generator_loss'] for x in self.outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in self.outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in self.outputs]).mean()
        
        # Update best metrics if current SSIM is better
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
            self.best_psnr = avg_psnr
            self.best_epoch = self.current_epoch
        
        # Log epoch metrics
        self.log('val/epoch_generator_loss', avg_gen_loss)
        self.log('val/epoch_psnr', avg_psnr)
        self.log('val/epoch_ssim', avg_ssim)
        
        # Print validation summary
        print(f"\nValidation Epoch {self.current_epoch} Summary:")
        print(f"Average Generator Loss: {avg_gen_loss:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("\nBest Results So Far:")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"Best PSNR: {self.best_psnr:.2f}")
        self.outputs.clear()
           
    def configure_optimizers(self):
        """Configure Adam optimizers for both generator and discriminator"""
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            eps=1e-8
        )
        
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            eps=1e-8
        )
        
        return [opt_g, opt_d]
    
class SRDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for handling SR dataset loading and preparation"""
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def setup(self, stage: Optional[str] = None):
        """Initialize train and validation datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = SRDataset(self.data_dir, split='train')
            self.val_dataset = SRDataset(self.data_dir, split='val')

    def train_dataloader(self):
        """Create training data loader with shuffling"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Create validation data loader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers)

def train_medsrgan():
    """
    Main training function for MedSRGAN.
    Sets up model, data, and training configuration with best practices.
    """
    model = MedSRGAN(lr=2e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop2', batch_size=1)
    
    # Configure checkpointing based on SSIM metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',
        filename='medsrgan-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
        save_top_k=1,
        mode='max',
    )
    
    # Configure training with mixed precision and monitoring
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',
        devices=1,
        precision=16,  # Use mixed precision training
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('logs', name='medsrgan_runs'),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    
    trainer.fit(model, datamodule)
    
    # Print final training summary
    print("\nTraining Completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best epoch: {model.best_epoch}")
    print(f"Best SSIM: {model.best_ssim:.4f}")
    print(f"Best PSNR: {model.best_psnr:.2f}")

if __name__ == "__main__":
    train_medsrgan()