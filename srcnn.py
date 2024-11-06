import pytorch_lightning as pl
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from typing import Optional, Tuple
import monai.losses

class SRDataset(Dataset):
    """
    Dataset class for Super-Resolution training data.
    Loads paired low-resolution and high-resolution grayscale images.
    """
    def __init__(self, root_dir, split='train'):
        # Define paths for low and high resolution image directories
        self.low_res_dir = os.path.join(root_dir, split, 'low_res')
        self.high_res_dir = os.path.join(root_dir, split, 'high_res')
        # Get list of all PNG images in the directory
        self.images = [f for f in os.listdir(self.low_res_dir) if f.endswith('.png')]
        
        # Transform to convert PIL images to tensors
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load and return a pair of low-res and high-res images.
        Returns:
            tuple: (low_resolution_tensor, high_resolution_tensor)
        """
        img_name = self.images[idx]
        
        # Load grayscale images
        low_res = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')
        high_res = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')
        
        # Convert to tensors (normalized to [0,1])
        low_res = self.transform(low_res)
        high_res = self.transform(high_res)
        
        return low_res, high_res
    
class SRCNN(pl.LightningModule):
    """
    Super-Resolution Convolutional Neural Network (SRCNN) implementation.
    Architecture consists of three layers:
    1. Feature extraction (9x9 conv)
    2. Non-linear mapping (5x5 conv)
    3. Reconstruction (5x5 conv)
    """
    def __init__(self, lr: float = 1e-4, target_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.outputs = []
        self.lr = lr
        self.target_size = target_size
        
        # Layer 1: Extract patch features using 9x9 kernels
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(True)
        )
        
        # Layer 2: Non-linear mapping between patches using 5x5 kernels
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(True)
        )
        
        # Layer 3: Final reconstruction using 5x5 kernels
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        
        self._initialize_weights()
        self.loss = nn.MSELoss()

        # Track best metrics during training
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0
        
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using specific initialization schemes:
        - Conv layers: Normal distribution with variance based on channel size
        - Reconstruction layer: Normal distribution with small variance
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 
                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        
        # Special initialization for reconstruction layer
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    def forward(self, x):
        """
        Forward pass of SRCNN:
        1. Upscale input using bicubic interpolation
        2. Extract features
        3. Map features
        4. Reconstruct high-res image
        """
        x = F.interpolate(x, size=self.target_size, mode='bicubic', align_corners=False)
                          
        out = self.features(x)
        out = self.map(out)
        return self.reconstruction(out)

    def training_step(self, batch, batch_idx):
        """Single training step"""
        low_res, high_res = batch
        sr_image = self(low_res)
        loss = self.loss(sr_image, high_res)
        self.log('train_loss', loss)
        return loss

    def calculate_ssim(self, img1, img2):
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        Uses a 11x11 Gaussian window and the standard SSIM formula.
        
        Args:
            img1, img2: Image tensors to compare
        Returns:
            float: SSIM value between -1 and 1 (1 indicates identical images)
        """
        C1 = (0.01 * 1.0) ** 2  # Constants for numerical stability
        C2 = (0.03 * 1.0) ** 2

        # Calculate means using average pooling
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2

        # Calculate SSIM using standard formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        return ssim_map.mean()

    def validation_step(self, batch, batch_idx):
        """
        Validation step computing loss and image quality metrics (PSNR, SSIM).
        Also saves validation image samples every 10 epochs.
        """
        low_res, high_res = batch
        sr_image = self(low_res)
        loss = self.loss(sr_image, high_res)
        
        # Calculate image quality metrics
        with torch.no_grad():
            # PSNR calculation
            mse = F.mse_loss(sr_image, high_res)
            psnr = 10 * torch.log10(1.0 / mse)
            
            # SSIM calculation
            ssim_value = self.calculate_ssim(sr_image, high_res)
            
        # Log metrics
        self.log_dict({
            'val/val_loss': loss,
            'val/psnr': psnr,
            'val/ssim': ssim_value
        }, prog_bar=True, sync_dist=True)
        
        # Save validation image samples every 10 epochs
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            os.makedirs('validation_images', exist_ok=True)
            
            # Create comparison grid: Low-res → Super-res → High-res
            comparison = torch.cat([
                F.interpolate(low_res, size=high_res.shape[-2:], mode='nearest'),
                sr_image,
                high_res
            ], dim=0)
            
            save_image(
                comparison,
                f'validation_images/epoch_{self.current_epoch+1}.png',
                nrow=len(low_res),
                normalize=True
            )
        
        # Store metrics for epoch-end averaging
        loss = {
            'val_loss': loss,
            'psnr': psnr,
            'ssim': ssim_value
        }
        
        self.outputs.append(loss)
            
        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log average metrics for the validation epoch.
        Updates best metrics if current results are better.
        """
        # Calculate average metrics
        avg_gen_loss = torch.stack([x['val_loss'] for x in self.outputs]).mean()
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
        """
        Configure optimization strategy:
        - Adam optimizer with specified learning rate
        - ReduceLROnPlateau scheduler that reduces LR when loss plateaus
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/epoch_generator_loss"
            }
        }

class SRDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling Super-Resolution datasets.
    Manages train and validation data loading.
    """
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
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
        """Create training data loader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Create validation data loader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers)

def train_srcnn():
    """
    Main training function for SRCNN model.
    Sets up model, data, and training configuration using PyTorch Lightning.
    """
    # Initialize model and data module
    model = SRCNN(lr=1e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop2', batch_size=6)
        
    # Setup model checkpointing based on SSIM metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',
        filename='srcnn-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
        save_top_k=1,  # Save only the best model
        mode='max',
    )
    
    # Configure trainer with mixed precision training and automatic hardware selection
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',
        devices=1,
        precision=16,  # Use mixed precision training
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('logs', name='srcnn_runs'),
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
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_srcnn()