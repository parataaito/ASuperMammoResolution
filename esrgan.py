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
    Loads and processes paired low-resolution and high-resolution grayscale images."""
    
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

class DenseBlock(nn.Module):
    """Dense Block for RRDB (Residual in Residual Dense Block).
    Features dense connections where each layer receives feature maps from all preceding layers."""
    
    def __init__(self, channels, growth_channels=32):
        """
        Args:
            channels: Number of input channels
            growth_channels: Growth rate for dense connections (default: 32)
        """
        super().__init__()
        # Series of convolutions with dense connections
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.beta = 0.2  # Scaling factor for residual connection

    def forward(self, x):
        """Implements dense connections where each layer gets input from all previous layers."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.beta + x  # Residual scaling

class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB).
    Contains three dense blocks with residual connections."""
    
    def __init__(self, channels):
        super().__init__()
        self.dense1 = DenseBlock(channels)
        self.dense2 = DenseBlock(channels)
        self.dense3 = DenseBlock(channels)
        self.beta = 0.2  # Scaling factor for residual connection

    def forward(self, x):
        """Chain of three dense blocks with residual connection."""
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * self.beta + x  # Residual scaling

class RRDBNet(nn.Module):
    """Enhanced SRGAN Generator architecture using Residual in Residual Dense Blocks (RRDB).
    Provides superior feature extraction and preservation compared to standard ResNet blocks."""
    
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=23, scale_factor=8):
        super().__init__()
        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # Chain of RRDB blocks for deep feature extraction
        self.body = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Upsampling blocks using pixel shuffle (efficient sub-pixel convolution)
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # 2x upscaling
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # 2x upscaling
            nn.LeakyReLU(0.2, True)
        )
        
        # Final reconstruction layer
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        """Forward pass with multiple residual connections."""
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # Global residual learning
        out = self.upsampling(feat)
        out = self.conv_last(out)
        return out

class Discriminator(nn.Module):
    """VGG-style discriminator with spectral normalization.
    Uses a series of strided convolutions to classify real vs generated images."""
    
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, batch_norm=True):
            """Helper function to create a discriminator block with optional batch norm."""
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        # Feature extraction layers
        self.features = nn.Sequential(
            *discriminator_block(1, 64, 2, False),  # No batch norm in first layer
            *discriminator_block(64, 128, 2),
            *discriminator_block(128, 256, 2),
            *discriminator_block(256, 512, 2),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1)  # Single output for binary classification
        )

    def forward(self, x):
        """Extract features and classify."""
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)  # Flatten features
        return self.classifier(feat)

class ESRGAN(pl.LightningModule):
    """PyTorch Lightning implementation of ESRGAN (Enhanced SRGAN).
    Features relativistic average GAN loss and perceptual loss."""
    
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.outputs = []
        self.automatic_optimization = False  # Manual optimization for separate G/D updates
        
        self.generator = RRDBNet()
        self.discriminator = Discriminator()
        self.lr = lr
        
        # Loss functions
        self.l1_loss = nn.L1Loss()  # Content loss
        self.perceptual_loss = monai.losses.PerceptualLoss(
            spatial_dims=2,
            network_type="vgg",
            is_fake_3d=False,
            pretrained=True
        )
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights for generator training
        self.content_weight = 0.01
        self.perceptual_weight = 1
        self.adversarial_weight = 0.005 
        
        # Best metric tracking
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0

    def compute_discriminator_adversarial_loss(self, hr_logits, sr_logits):
        """Implements relativistic average discriminator loss.
        RaD tries to maximize the difference between real and fake samples."""
        # Calculate relativistic average predictions
        avg_fake = sr_logits.mean()
        avg_real = hr_logits.mean()
        
        D_Ra_real = torch.sigmoid(hr_logits - avg_fake)
        D_Ra_fake = torch.sigmoid(sr_logits - avg_real)
        
        # Compute binary cross entropy losses
        real_loss = -torch.mean(torch.log(D_Ra_real + 1e-8))
        fake_loss = -torch.mean(torch.log(1 - D_Ra_fake + 1e-8))
        
        return real_loss + fake_loss
    
    def compute_generator_adversarial_loss(self, hr_logits, sr_logits):
        """Implements relativistic average generator loss.
        RaG tries to minimize the difference between real and fake samples."""
        # Calculate relativistic average predictions
        avg_fake = sr_logits.mean()
        avg_real = hr_logits.mean()
        
        D_Ra_real = torch.sigmoid(hr_logits - avg_fake)
        D_Ra_fake = torch.sigmoid(sr_logits - avg_real)
        
        # Compute binary cross entropy losses (reversed from discriminator)
        real_loss = -torch.mean(torch.log(1 - D_Ra_real + 1e-8))
        fake_loss = -torch.mean(torch.log(D_Ra_fake + 1e-8))
        
        return real_loss + fake_loss

    def training_step(self, batch, batch_idx):
        """Performs alternating generator and discriminator updates."""
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
        
        # Generator update
        opt_g.zero_grad()
        sr_imgs = self(lr_imgs)
        g_loss, content_loss, perceptual_loss, adversarial_loss = self.compute_generator_loss(hr_imgs, sr_imgs)
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Discriminator update
        opt_d.zero_grad()
        d_loss = self.compute_discriminator_loss(hr_imgs, sr_imgs)
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Log training metrics
        self.log_dict({
            'g_loss': g_loss,
            'd_loss': d_loss,
            'content_loss': content_loss,
            'adversarial_loss': adversarial_loss,
            'perceptual_loss': perceptual_loss
        }, prog_bar=True)

    def compute_discriminator_loss(self, hr_imgs, sr_imgs):
        """Computes the relativistic average discriminator loss."""
        hr_logits = self.discriminator(hr_imgs)
        sr_logits = self.discriminator(sr_imgs.detach())  # Detach to avoid computing gradients for generator
        return self.compute_discriminator_adversarial_loss(hr_logits, sr_logits)

    def compute_generator_loss(self, hr_imgs, sr_imgs):
        """Computes the full generator loss including content, perceptual, and adversarial terms."""
        hr_logits = self.discriminator(hr_imgs)
        sr_logits = self.discriminator(sr_imgs)
        
        # Calculate individual loss components
        content_loss = self.l1_loss(sr_imgs, hr_imgs)
        perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
        adversarial_loss = self.compute_generator_adversarial_loss(hr_logits, sr_logits)
        
        # Combine losses with weights
        g_loss = (self.content_weight * content_loss + 
                 self.perceptual_weight * perceptual_loss +
                 self.adversarial_weight * adversarial_loss)
                   
        return g_loss, content_loss, perceptual_loss, adversarial_loss

    def calculate_ssim(self, img1, img2):
        """Calculates Structural Similarity Index (SSIM) between two images."""
        C1 = (0.01 * 1.0) ** 2  # Constants for numerical stability
        C2 = (0.03 * 1.0) ** 2

        # Calculate mean values using average pooling
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
        
        # Calculate variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2

        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        return ssim_map.mean()

    def validation_step(self, batch, batch_idx):
        """Performs validation step and computes quality metrics (PSNR, SSIM)."""
        lr_imgs, hr_imgs = batch
        
        # Generate SR images and compute losses
        with torch.no_grad():
            sr_imgs = self(lr_imgs)
            
            # Calculate losses
            g_loss, content_loss, perceptual_loss, adversarial_loss = self.compute_generator_loss(hr_imgs, sr_imgs)
            d_loss = self.compute_discriminator_loss(hr_imgs, sr_imgs)


        # Calculate image quality metrics
        with torch.no_grad():
            # Calculate Peak Signal-to-Noise Ratio (PSNR)
            mse = F.mse_loss(sr_imgs, hr_imgs)
            psnr = 10 * torch.log10(1.0 / mse)
            
            # Calculate Structural Similarity Index (SSIM)
            ssim_value = self.calculate_ssim(sr_imgs, hr_imgs)        
        
        # Log all validation metrics
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
            
            # Create comparison grid: LR (upscaled) | SR (generated) | HR (ground truth)
            comparison = torch.cat([
                F.interpolate(lr_imgs, size=hr_imgs.shape[-2:], mode='nearest'),
                sr_imgs,
                hr_imgs
            ], dim=0)
            
            # Save the comparison grid
            save_image(
                comparison,
                f'validation_images/epoch_{self.current_epoch+1}.png',
                nrow=len(lr_imgs),
                normalize=True
            )
            
        # Store metrics for epoch-end averaging
        loss = {
            'generator_loss': g_loss,
            'psnr': psnr,
            'ssim': ssim_value
        }
        
        self.outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """Computes and logs epoch-level validation metrics and tracks best model performance."""
        # Calculate average metrics across all validation batches
        avg_gen_loss = torch.stack([x['generator_loss'] for x in self.outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in self.outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in self.outputs]).mean()
        
        # Update best metrics if current results are better
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
            self.best_psnr = avg_psnr
            self.best_epoch = self.current_epoch
        
        # Log epoch-level metrics
        self.log('val/epoch_generator_loss', avg_gen_loss)
        self.log('val/epoch_psnr', avg_psnr)
        self.log('val/epoch_ssim', avg_ssim)
        
        # Print comprehensive validation summary
        print(f"\nValidation Epoch {self.current_epoch} Summary:")
        print(f"Average Generator Loss: {avg_gen_loss:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("\nBest Results So Far:")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"Best PSNR: {self.best_psnr:.2f}")
        
        self.outputs.clear()  # Free memory
 
    def configure_optimizers(self):
        """Configures Adam optimizers for both generator and discriminator."""
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.9, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return [opt_g, opt_d]

class SRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling Super Resolution dataset.
    Manages train and validation data loaders with efficient multi-process data loading."""
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 11):
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
        """Creates train and validation datasets on setup."""
        if stage == "fit" or stage is None:
            self.train_dataset = SRDataset(self.data_dir, split='train')
            self.val_dataset = SRDataset(self.data_dir, split='val')

    def train_dataloader(self):
        """Returns the training data loader with shuffling enabled."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True  # Enable shuffling for training
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

def train_esrgan():
    """Main training function that sets up and executes the ESRGAN training process."""
    
    # Initialize model and data module
    model = ESRGAN(lr=1e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop', batch_size=1)
    
    # Configure model checkpointing based on SSIM
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',      # Monitor SSIM for model saving
        filename='esrgan-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
        save_top_k=1,                  # Save only the best model
        mode='max',                    # Higher SSIM is better
    )
    
    # Configure training parameters
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',            # Automatically detect GPU/CPU
        devices=1,
        precision=16,                  # Use mixed precision training
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('logs', name='esrgan_runs'),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
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
    train_esrgan()