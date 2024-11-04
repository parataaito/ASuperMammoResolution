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
    def __init__(self, root_dir, split='train'):
        self.low_res_dir = os.path.join(root_dir, split, 'low_res')
        self.high_res_dir = os.path.join(root_dir, split, 'high_res')
        self.images = [f for f in os.listdir(self.low_res_dir) if f.endswith('.png')]
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load images
        low_res = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')  # Convert to grayscale
        high_res = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')  # Convert to grayscale
        
        # Convert to tensors
        low_res = self.transform(low_res)
        high_res = self.transform(high_res)
        
        return low_res, high_res
   
class ResidualBlock(nn.Module):
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
        out += residual
        return out

class Generator(nn.Module):
    def __init__(self, scale_factor=8):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks
        res_blocks = []
        for _ in range(16):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Post-residual convolution
        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Three upsampling blocks for 8x upscaling (2*2*2 = 8)
        self.upsampling = nn.Sequential(
            # First 2x upscale
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            
            # Second 2x upscale
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        
        # Output convolution
        self.conv_output = nn.Conv2d(64, 1, kernel_size=9, padding=4)
        
    def forward(self, x):
        # Input shape: [batch, 1, 220, 175]
        out = self.conv_input(x)  # [batch, 64, 220, 175]
        res = self.res_blocks(out)
        out = self.conv_mid(res)
        out = out + self.conv_input(x)  # Skip connection
        
        # Three 2x upscaling steps: 220->440->880->1760 and 175->350->700->1400
        out = self.upsampling(out)
        return self.conv_output(out)  # Final output: [batch, 1, 1760, 1400]
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
            )
        
        self.model = nn.Sequential(
            conv_block(1, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 2),
            conv_block(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

class SRGAN(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.outputs = []
        self.automatic_optimization = False
        
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lr = lr
        
        # Loss functions
        self.content_loss = nn.MSELoss()
        
        # MONAI perceptual loss
        self.perceptual_loss = monai.losses.PerceptualLoss(
            spatial_dims=2,
            network_type="squeeze",
            is_fake_3d=False,
            pretrained=True
        )
        
        # Loss weights
        self.content_weight = 1
        self.perceptual_weight = 0.1
        self.adversarial_weight = 0.001
        
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0

    def forward(self, x):
        return self.generator(x)

    def generator_adversarial_loss(self, discriminator_predictions):
        # Add small epsilon to prevent log(0)
        return -torch.mean(torch.log(discriminator_predictions + 1e-10))

    def discriminator_adversarial_loss(self, real_preds, fake_preds):
        # Add small epsilon to prevent log(0)
        real_loss = -torch.mean(torch.log(real_preds + 1e-10))
        fake_loss = -torch.mean(torch.log(1 - fake_preds + 1e-10))
        return (real_loss + fake_loss) * 0.5

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
                   
        # Train Generator
        opt_g.zero_grad(set_to_none=True)
        
        # Generate SR images
        sr_imgs = self(lr_imgs)
            
        fake_preds_g = self.discriminator(sr_imgs)
        
        # Calculate generator losses
        content_loss = self.content_loss(sr_imgs, hr_imgs)
        perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
        adversarial_loss = self.generator_adversarial_loss(fake_preds_g)
        
        gen_loss = (self.content_weight * content_loss + 
                   self.perceptual_weight * perceptual_loss +
                   self.adversarial_weight * adversarial_loss)
        
        self.manual_backward(gen_loss)
        opt_g.step()
        
        # # Clear memory after generator update
        # fake_preds_g = None
        # gen_loss = None
        
        # Train Discriminator
        opt_d.zero_grad(set_to_none=True)
        
        # Generate new SR images for discriminator training
        with torch.no_grad():
            sr_imgs_d = self(lr_imgs)
        
        # Get new predictions for discriminator training
        real_preds = self.discriminator(hr_imgs)
        fake_preds_d = self.discriminator(sr_imgs_d)  # No need for detach() since we used torch.no_grad()
        
        # Calculate discriminator loss
        d_loss = self.discriminator_adversarial_loss(real_preds, fake_preds_d)
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Log losses
        self.log_dict({
            'g_loss': gen_loss,
            'd_loss': d_loss,
            'content_loss': content_loss,
            'perceptual_loss': perceptual_loss,
            'adversarial_loss': adversarial_loss,
        }, prog_bar=True)

    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM between two images
        Args:
            img1: First image tensor
            img2: Second image tensor
        Returns:
            SSIM value
        """
        # Constants for numerical stability
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        # Compute means
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

        # Compute squares of means
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2

        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        # Return mean SSIM
        return ssim_map.mean()

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        
        # Generate SR images
        with torch.no_grad():
            sr_imgs = self(lr_imgs)
            
            # Get discriminator predictions
            fake_preds = self.discriminator(sr_imgs)
            real_preds = self.discriminator(hr_imgs)

            # Calculate generator losses
            content_loss = self.content_loss(sr_imgs, hr_imgs)
            perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
            adversarial_loss = self.generator_adversarial_loss(fake_preds)
            
            g_loss = (self.content_weight * content_loss + 
                    self.perceptual_weight * perceptual_loss +
                    self.adversarial_weight * adversarial_loss)

            # Calculate discriminator loss
            d_loss = self.discriminator_adversarial_loss(real_preds, fake_preds)
        
        # Calculate metrics
        with torch.no_grad():
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = F.mse_loss(sr_imgs, hr_imgs)
            psnr = 10 * torch.log10(1.0 / mse)
            
            # SSIM (Structural Similarity Index)
            ssim_value = self.calculate_ssim(sr_imgs, hr_imgs)
        
        
        self.log_dict({
            'val/generator_loss': g_loss,
            'val/discriminator_loss': d_loss,
            'val/content_loss': content_loss,
            'val/adversarial_loss': adversarial_loss,
            'val/perceptual_loss': perceptual_loss,
            'val/psnr': psnr,
            'val/ssim': ssim_value
        }, prog_bar=True, sync_dist=True)
        
        
        # Save validation images periodically
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            # Create validation directory if it doesn't exist
            os.makedirs('validation_images', exist_ok=True)
            
            # Save grid of images (LR, SR, HR)
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
            
        # Store metrics for epoch end averaging
        loss = {
            'generator_loss': g_loss,
            'psnr': psnr,
            'ssim': ssim_value
        }
        
        self.outputs.append(loss)
        
        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        return [opt_g, opt_d]

    def on_validation_epoch_end(self):
        # Stack all the metrics from validation steps
        avg_gen_loss = torch.stack([x['generator_loss'] for x in self.outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in self.outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in self.outputs]).mean()
        
        # Update best metrics
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
            self.best_psnr = avg_psnr
            self.best_epoch = self.current_epoch
        
        # Log epoch-level metrics
        self.log('val/epoch_generator_loss', avg_gen_loss)
        self.log('val/epoch_psnr', avg_psnr)
        self.log('val/epoch_ssim', avg_ssim)
        
        # Print summary with best results
        print(f"\nValidation Epoch {self.current_epoch} Summary:")
        print(f"Average Generator Loss: {avg_gen_loss:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("\nBest Results So Far:")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print(f"Best PSNR: {self.best_psnr:.2f}")
        self.outputs.clear()  # free memory
 
class SRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SRDataset(self.data_dir, split='train')
            self.val_dataset = SRDataset(self.data_dir, split='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers)

def train_srgan():
    model = SRGAN(lr=1e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop2', batch_size=8)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',
        filename='srgan-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
        save_top_k=1,  # Only save the best model
        mode='max',
    )
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',
        devices=1,
        precision=16,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('logs', name='srgan_runs'),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    
    trainer.fit(model, datamodule)
    
    print("\nTraining Completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best epoch: {model.best_epoch}")
    print(f"Best SSIM: {model.best_ssim:.4f}")
    print(f"Best PSNR: {model.best_psnr:.2f}")
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_srgan()