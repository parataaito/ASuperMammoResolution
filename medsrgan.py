import pytorch_lightning as pl
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
    def __init__(self, root_dir, split='train'):
        self.low_res_dir = os.path.join(root_dir, split, 'low_res')
        self.high_res_dir = os.path.join(root_dir, split, 'high_res')
        self.images = [f for f in os.listdir(self.low_res_dir) if f.endswith('.png')]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def normalize_minmax(self, tensor):
        """Normalize tensor to [0,1] range"""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:  # Avoid division by zero
            return (tensor - min_val) / (max_val - min_val)
        return tensor

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        low_res_orig = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')
        high_res_orig = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')
        
        low_res = self.transform(low_res_orig)
        high_res = self.transform(high_res_orig)
                
        # # Degradation
        # # Apply gaussian noise
        # if np.random.rand() > 0.5:
        #     low_res = self.apply_gaussian_noise(low_res)
        
        # Apply min-max normalization
        low_res = self.normalize_minmax(low_res)
        high_res = self.normalize_minmax(high_res)

        return low_res, high_res

    def apply_gaussian_noise(self, low_res, mean=0.0, std=0.1):
        noise = torch.randn_like(low_res) * std + mean
        low_res = low_res + noise
        return low_res

class D_Block(nn.Module):
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
    def __init__(self, in_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ = self.layer1(x)
        x__ = self.layer2(x_)
        x = x__ * x_ + x
        return x

class ShortResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.ModuleList([RWMAB(in_channels) for _ in range(16)])

    def forward(self, x):
        x_ = x.clone()
        for layer in self.layers:
            x_ = layer(x_)
        return x_ + x

class Generator(nn.Module):
    def __init__(self, in_channels=1, blocks=8):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1)
        self.short_blocks = nn.ModuleList(
            [ShortResidualBlock(64) for _ in range(blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)
        
        # Modified for 8x upsampling (3 pixel shuffle layers)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, (3, 3), stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, in_channels, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x_ = x.clone()

        for layer in self.short_blocks:
            x_ = layer(x_)

        x = torch.cat([self.conv2(x_), x], dim=1)
        x = self.conv3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1), 
            nn.LeakyReLU()
        )

        # Path 1: Reduce spatial dimensions more aggressively
        self.block_1_1 = D_Block(64, 64, stride=2)    # 1024 -> 512         512 -> 256
        self.block_1_2 = D_Block(64, 128, stride=2)   # 512 -> 256          256 -> 128
        self.block_1_3 = D_Block(128, 128, stride=1)  # Size stays at 256   128 -> 128

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), stride=1, padding=1), 
            nn.LeakyReLU()
        )

        # Path 2: Reduce to match Path 1
        self.block_2_1 = D_Block(64, 64, stride=2)    # 1024 -> 512         512 -> 256
        self.block_2_2 = D_Block(64, 128, stride=2)   # 512 -> 256          256 -> 128

        # Combined path
        self.block3 = D_Block(256, 256, stride=1)
        self.block4 = D_Block(256, 256, stride=2)     # 256 -> 128          128 -> 64
        self.block5 = D_Block(256, 512, stride=1)
        self.block6 = D_Block(512, 512, stride=2)     # 128 -> 64           64 -> 32
        self.block7 = D_Block(512, 1024, stride=2)    # 64 -> 32            32 -> 16
        self.block8 = D_Block(1024, 1024, stride=2)   # 32 -> 16            16 -> 8

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def get_features(self, x):
        """Return intermediate features for adversarial feature loss"""
        features = []
        
        # Path 1
        x_1 = self.conv_1_1(x)
        features.append(x_1)
        
        x_1 = self.block_1_1(x_1)
        features.append(x_1)
        
        x_1 = self.block_1_2(x_1)
        x_1 = self.block_1_3(x_1)
        features.append(x_1)
        
        # Path 2
        x_2 = self.conv_2_1(x)
        x_2 = self.block_2_1(x_2)
        x_2 = self.block_2_2(x_2)
        features.append(x_2)
        
        # Combined path
        x = torch.cat([x_1, x_2], dim=1)
        x = self.block3(x)
        features.append(x)
        
        return features
    
    def forward(self, x, get_features=False):
        if get_features:
            return self.get_features(x)
        
        # Path 1
        x_1 = self.conv_1_1(x)
        x_1 = self.block_1_1(x_1)
        x_1 = self.block_1_2(x_1)
        x_1 = self.block_1_3(x_1)  # Should be [B, 128, 256, 256]   [B, 128, 128, 128]
        
        # Path 2
        x_2 = self.conv_2_1(x)
        x_2 = self.block_2_1(x_2)
        x_2 = self.block_2_2(x_2)  # Should be [B, 128, 256, 256]   [B, 128, 256, 256]

        # Combine paths
        x = torch.cat([x_1, x_2], dim=1)  # Should be [B, 256, 256, 256]
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        return self.sigmoid(x)

class MedSRGAN(pl.LightningModule):
    def __init__(self, lr: float = 2e-4):  # Increased learning rate
        super().__init__()
        self.outputs = []
        self.automatic_optimization = False
        
        self.generator = Generator(in_channels=1)
        self.discriminator = Discriminator(in_channels=1)
        self.lr = lr
        
        # Initialize weights properly
        self.generator.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)
        
        # Loss functions with adjusted weights
        self.content_l1_loss = nn.L1Loss()
        self.content_vgg_loss = monai.losses.PerceptualLoss(
            spatial_dims=2,
            network_type="vgg",
            is_fake_3d=False,
            pretrained=True
        )
        
        # Adjusted loss weights
        # self.l1_weight = 0
        # self.content_weight = 1.0
        # self.perceptual_weight = 0.1
        # self.adversarial_weight = 0.001
        self.content_l1_weight = 0.01
        self.content_vgg_weight = 1
        self.content_weight = 1.0
        
        self.adversarial_weight = 0.05
        
        self.feature_weights = [1/2, 1/4, 1/8, 1/16, 1/16]
        self.adversarial_feature_weight = 0.005
        
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0
        
    def forward(self, x):
        return self.generator(x)
            
    def _init_weights(self, m):
        """Initialize network weights properly"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def compute_discriminator_loss(self, hr_imgs, sr_imgs):
        # Get discriminator predictions
        hr_logits = self.discriminator(hr_imgs)
        sr_logits = self.discriminator(sr_imgs.detach())
        
        # Add noise to labels for label smoothing
        real_labels = torch.rand_like(hr_logits) * 0.1 + 0.9  # Random between 0.9 and 1.0
        fake_labels = torch.rand_like(sr_logits) * 0.1        # Random between 0.0 and 0.1
        
        # Compute losses with label smoothing
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
        Compute adversarial feature loss according to equation (8)
        Args:
            lr_imgs: Low resolution images
            hr_imgs: High resolution images
            sr_imgs: Super-resolution (generated) images
        Returns:
            Adversarial feature loss
        """
        # Get features from discriminator for each image type
        hr_features = self.discriminator.get_features(hr_imgs)
        lr_features = self.discriminator.get_features(lr_imgs)
        sr_features = self.discriminator.get_features(sr_imgs)
        
        feature_loss = 0.0
        # Compute weighted MSE for each feature layer
        for i, (w, hr_feat, lr_feat, sr_feat) in enumerate(zip(
            self.feature_weights, hr_features, lr_features, sr_features)):
            # Interpolate LR features to match HR/SR spatial dimensions
            lr_feat_upscaled = F.interpolate(
                lr_feat,
                size=hr_feat.shape[2:],  # Get target spatial dimensions
                mode='bilinear',
                align_corners=False
            )
            
            # Compute MSE between LR and HR features
            mse_lr_hr = F.mse_loss(lr_feat_upscaled, hr_feat)
            # Compute MSE between LR and SR features
            mse_lr_sr = F.mse_loss(lr_feat_upscaled, sr_feat)
            # Add weighted loss for this layer
            feature_loss += w * (mse_lr_hr + mse_lr_sr)
            
            # Log individual layer losses for debugging
            self.log(f'feature_loss_layer_{i}', w * (mse_lr_hr + mse_lr_sr))
            
        return feature_loss
    
    def compute_generator_loss(self, lr_imgs, sr_imgs, hr_imgs):
        # Content loss (L1)
        content_loss_l1 = self.content_l1_loss(sr_imgs, hr_imgs)
        content_loss_vgg = self.content_vgg_loss(sr_imgs, hr_imgs)
        content_loss = self.content_l1_weight * content_loss_l1 + self.content_vgg_weight * content_loss_vgg
                
        # Adversarial loss
        fake_logits = self.discriminator(sr_imgs)
        adversarial_loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits)
        )

        # Perceptual loss
        # Add adversarial feature loss
        adversarial_feature_loss = self.compute_adversarial_feature_loss(lr_imgs, hr_imgs, sr_imgs)

        # Combine losses with weights
        total_loss = (
            self.content_weight * content_loss +
            self.adversarial_feature_weight * adversarial_feature_loss +
            self.adversarial_weight * adversarial_loss
        )
        
        return total_loss, content_loss, adversarial_loss, adversarial_feature_loss

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
        
        # Train discriminator first
        opt_d.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Generate super-resolution images
        with torch.cuda.amp.autocast():  # Use mixed precision
            sr_imgs = self(lr_imgs)
            d_loss = self.compute_discriminator_loss(hr_imgs, sr_imgs)
        
        # Backward pass with gradient clipping
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        opt_d.step()
        
        # Train generator
        opt_g.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            sr_imgs = self(lr_imgs)
            g_loss, content_loss, adv_loss, percep_loss = self.compute_generator_loss(lr_imgs, sr_imgs, hr_imgs)
        
        # Backward pass with gradient clipping
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        opt_g.step()
        
        # Log losses
        self.log_dict({
            'g_loss': g_loss,
            'd_loss': d_loss,
            'content_loss': content_loss,
            'adversarial_loss': adv_loss,
            'perceptual_loss': percep_loss
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
        with torch.cuda.amp.autocast():
            sr_imgs = self(lr_imgs)
            
            # Calculate losses
            g_loss, content_loss, adv_loss, percep_loss = self.compute_generator_loss(lr_imgs, sr_imgs, hr_imgs)
            d_loss = self.compute_discriminator_loss(hr_imgs, sr_imgs)
        
        # Calculate metrics
        with torch.no_grad():
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = F.mse_loss(sr_imgs, hr_imgs)
            psnr = 10 * torch.log10(1.0 / mse)
            
            # SSIM (Structural Similarity Index)
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
           
    def configure_optimizers(self):
        # Use different learning rates for generator and discriminator
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            eps=1e-8
        )
        
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,  # Slower learning rate for discriminator
            betas=(0.5, 0.999),
            eps=1e-8
        )
        
        return [opt_g, opt_d]
    
class SRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 1):
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

def train_medsrgan():
    model = MedSRGAN(lr=2e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop2', batch_size=1)  # Smaller batch size
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',
        filename='medsrgan-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
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
        logger=pl.loggers.TensorBoardLogger('logs', name='medsrgan_runs'),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    
    trainer.fit(model, datamodule)
    
    print("\nTraining Completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best epoch: {model.best_epoch}")
    print(f"Best SSIM: {model.best_ssim:.4f}")
    print(f"Best PSNR: {model.best_psnr:.2f}")

if __name__ == "__main__":
    train_medsrgan()