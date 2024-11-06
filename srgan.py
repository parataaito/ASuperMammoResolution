import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from typing import Optional, Tuple
import monai.losses

class SRDataset(Dataset):
    """Dataset class for Super Resolution training.
    Loads paired low-resolution and high-resolution grayscale images from specified directories."""
    
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: Base directory containing the dataset
            split: Dataset split ('train' or 'val')
        """
        self.low_res_dir = os.path.join(root_dir, split, 'low_res')
        self.high_res_dir = os.path.join(root_dir, split, 'high_res')
        self.images = [f for f in os.listdir(self.low_res_dir) if f.endswith('.png')]
        
        self.transform = transforms.Compose([
            transforms.ToTensor()  # Converts PIL Image to tensor and scales to [0,1]
        ])

    def __getitem__(self, idx):
        """Returns a pair of low-resolution and high-resolution images as tensors."""
        img_name = self.images[idx]
        
        # Load and convert images to grayscale (single channel)
        low_res = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')
        high_res = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')
        
        return self.transform(low_res), self.transform(high_res)
   
class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection.
    Architecture: Conv -> BN -> PReLU -> Conv -> BN -> Add Input"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual  # Skip connection

class Generator(nn.Module):
    """SRGAN Generator architecture with residual blocks and pixel shuffle upsampling.
    Performs 4x upscaling through two 2x pixel shuffle operations."""
    
    def __init__(self, scale_factor=8):
        super().__init__()
        
        # Initial feature extraction
        self.conv_input = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),  # Large kernel for better feature extraction
            nn.PReLU()
        )
        
        # Stack of residual blocks for deep feature extraction
        res_blocks = [ResidualBlock(64) for _ in range(16)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Bridge between residual blocks and upsampling
        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Progressive upsampling using pixel shuffle
        # Each block doubles the spatial dimensions
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2x upscaling
            nn.PReLU(),
            
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2x upscaling
            nn.PReLU(),
        )
        
        # Final reconstruction layer
        self.conv_output = nn.Conv2d(64, 1, kernel_size=9, padding=4)
        
    def forward(self, x):
        """
        Forward pass of generator.
        Input: [batch, 1, H, W]
        Output: [batch, 1, H*4, W*4]
        """
        initial_features = self.conv_input(x)
        res_features = self.res_blocks(initial_features)
        bridge = self.conv_mid(res_features)
        enhanced_features = bridge + initial_features  # Global skip connection
        upscaled = self.upsampling(enhanced_features)
        return self.conv_output(upscaled)
    
class Discriminator(nn.Module):
    """SRGAN Discriminator with strided convolutions for downsampling.
    Outputs probability of input being real vs generated."""
    
    def __init__(self):
        super().__init__()
        
        def conv_block(in_channels, out_channels, stride=1):
            """Helper function to create a convolution block with batch norm and LeakyReLU"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
            )
        
        # Progressive downsampling with increasing channels
        self.model = nn.Sequential(
            conv_block(1, 64, 2),      # H/2
            conv_block(64, 128, 2),    # H/4
            conv_block(128, 256, 2),   # H/8
            conv_block(256, 512, 2),   # H/16
            nn.AdaptiveAvgPool2d(1),   # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # Output probability
        )

class SRGAN(pl.LightningModule):
    """PyTorch Lightning implementation of SRGAN.
    Combines Generator and Discriminator training with perceptual loss."""
    
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.outputs = []
        self.automatic_optimization = False  # Manual optimization for separate G/D updates
        
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lr = lr
        
        # Loss functions
        self.content_loss = nn.MSELoss()  # Pixel-wise loss
        self.perceptual_loss = monai.losses.PerceptualLoss(
            spatial_dims=2,
            network_type="squeeze",
            is_fake_3d=False,
            pretrained=True
        )
        
        # Loss weights for generator training
        self.content_weight = 1
        self.perceptual_weight = 0.1
        self.adversarial_weight = 0.001
        
        # Best metric tracking
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0

    def generator_adversarial_loss(self, discriminator_predictions):
        """Computes generator adversarial loss using log probability."""
        return -torch.mean(torch.log(discriminator_predictions + 1e-10))

    def discriminator_adversarial_loss(self, real_preds, fake_preds):
        """Computes discriminator loss using binary cross entropy principle."""
        real_loss = -torch.mean(torch.log(real_preds + 1e-10))
        fake_loss = -torch.mean(torch.log(1 - fake_preds + 1e-10))
        return (real_loss + fake_loss) * 0.5

    def training_step(self, batch, batch_idx):
        """Performs a training step with generator and discriminator updates."""
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
                   
        # Generator training
        opt_g.zero_grad(set_to_none=True)
        sr_imgs = self(lr_imgs)
        fake_preds_g = self.discriminator(sr_imgs)
        
        # Compute generator losses
        content_loss = self.content_loss(sr_imgs, hr_imgs)
        perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
        adversarial_loss = self.generator_adversarial_loss(fake_preds_g)
        
        gen_loss = (self.content_weight * content_loss + 
                   self.perceptual_weight * perceptual_loss +
                   self.adversarial_weight * adversarial_loss)
        
        self.manual_backward(gen_loss)
        opt_g.step()
        
        # Discriminator training
        opt_d.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            sr_imgs_d = self(lr_imgs)
        
        real_preds = self.discriminator(hr_imgs)
        fake_preds_d = self.discriminator(sr_imgs_d)
        
        d_loss = self.discriminator_adversarial_loss(real_preds, fake_preds_d)
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Log training metrics
        self.log_dict({
            'g_loss': gen_loss,
            'd_loss': d_loss,
            'content_loss': content_loss,
            'perceptual_loss': perceptual_loss,
            'adversarial_loss': adversarial_loss,
        }, prog_bar=True)

    def calculate_ssim(self, img1, img2):
        """Calculates Structural Similarity Index (SSIM) between two images."""
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
        """Performs validation step and computes quality metrics (PSNR, SSIM)."""
        lr_imgs, hr_imgs = batch
        
        with torch.no_grad():
            # Generate SR images and compute losses
            sr_imgs = self(lr_imgs)
            fake_preds = self.discriminator(sr_imgs)
            real_preds = self.discriminator(hr_imgs)

            content_loss = self.content_loss(sr_imgs, hr_imgs)
            perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
            adversarial_loss = self.generator_adversarial_loss(fake_preds)
            
            g_loss = (self.content_weight * content_loss + 
                    self.perceptual_weight * perceptual_loss +
                    self.adversarial_weight * adversarial_loss)

            d_loss = self.discriminator_adversarial_loss(real_preds, fake_preds)
            
            # Calculate image quality metrics
            mse = F.mse_loss(sr_imgs, hr_imgs)
            psnr = 10 * torch.log10(1.0 / mse)
            ssim_value = self.calculate_ssim(sr_imgs, hr_imgs)
        
        # Log validation metrics
        self.log_dict({
            'val/generator_loss': g_loss,
            'val/discriminator_loss': d_loss,
            'val/content_loss': content_loss,
            'val/adversarial_loss': adversarial_loss,
            'val/perceptual_loss': perceptual_loss,
            'val/psnr': psnr,
            'val/ssim': ssim_value
        }, prog_bar=True, sync_dist=True)
        
        # Save validation image samples every 10 epochs
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

    def configure_optimizers(self):
        """Configures separate optimizers for generator and discriminator."""
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [opt_g, opt_d]

    def on_validation_epoch_end(self):
        """Computes and logs epoch-level validation metrics and tracks best model."""
        avg_gen_loss = torch.stack([x['generator_loss'] for x in self.outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in self.outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in self.outputs]).mean()
        
        # Update best metrics if current results are better
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
            self.best_psnr = avg_psnr
            self.best_epoch = self.current_epoch
        
        # Log epoch-level metrics
        self.log_dict({
            'val/epoch_generator_loss': avg_gen_loss,
            'val/epoch_psnr': avg_psnr,
            'val/epoch_ssim': avg_ssim
        })
        
        # Print validation summary
        print(f"\nValidation Epoch {self.current_epoch} Summary:")
        print(f"Average Generator Loss: {avg_gen_loss:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("\nBest Results So Far:")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"Best PSNR: {self.best_psnr:.2f}")
        self.outputs.clear()  # Clear stored outputs to free memory
 
class SRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling Super Resolution dataset.
    Manages train and validation data loaders."""
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        """
        Args:
            data_dir: Root directory containing the dataset
            batch_size: Number of samples per batch
            num_workers: Number of parallel workers for data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def setup(self, stage: Optional[str] = None):
        """Creates train and validation datasets.
        Called automatically by PyTorch Lightning."""
        if stage == "fit" or stage is None:
            self.train_dataset = SRDataset(self.data_dir, split='train')
            self.val_dataset = SRDataset(self.data_dir, split='val')

    def train_dataloader(self):
        """Returns the training data loader."""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True  # Shuffle training data for better generalization
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

def train_srgan():
    """Main training function for SRGAN model.
    Sets up model, data, and training parameters, then starts training."""
    
    # Initialize model and data module
    model = SRGAN(lr=1e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop2', batch_size=8)
    
    # Configure model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',  # Monitor SSIM for model saving
        filename='srgan-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
        save_top_k=1,  # Save only the best model
        mode='max',    # Higher SSIM is better
    )
    
    # Configure training parameters
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',     # Automatically detect GPU/CPU
        devices=1,              # Number of devices to use
        precision=16,           # Use mixed precision training for efficiency
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')  # Monitor learning rate
        ],
        logger=pl.loggers.TensorBoardLogger('logs', name='srgan_runs'),  # TensorBoard logging
        log_every_n_steps=10,          # Logging frequency
        check_val_every_n_epoch=1,     # Validation frequency
    )
    
    # Start training
    trainer.fit(model, datamodule)
    
    # Print final results
    print("\nTraining Completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best epoch: {model.best_epoch}")
    print(f"Best SSIM: {model.best_ssim:.4f}")
    print(f"Best PSNR: {model.best_psnr:.2f}")

if __name__ == "__main__":
    train_srgan()