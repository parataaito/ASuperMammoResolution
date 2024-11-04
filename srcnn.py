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
# import monai.transforms as transforms

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
    
class SRCNN(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, target_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.outputs = []
        self.lr = lr
        
        self.target_size = target_size
        
        # Feature extraction layer
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(True)
        )
        
        # Non-linear mapping layer
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(True)
        )
        
        # Reconstruction layer
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        
        self._initialize_weights()
        self.loss = nn.MSELoss()

        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_epoch = 0
        
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 
                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode='bicubic', align_corners=False)
                          
        out = self.features(x)
        out = self.map(out)
        return self.reconstruction(out)

    def training_step(self, batch, batch_idx):
        low_res, high_res = batch
        sr_image = self(low_res)
        loss = self.loss(sr_image, high_res)
        self.log('train_loss', loss)
        return loss

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
        low_res, high_res = batch
        sr_image = self(low_res)
        loss = self.loss(sr_image, high_res)
        
        # Calculate metrics
        with torch.no_grad():
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = F.mse_loss(sr_image, high_res)
            psnr = 10 * torch.log10(1.0 / mse)
            
            # SSIM (Structural Similarity Index)
            ssim_value = self.calculate_ssim(sr_image, high_res)
            
        self.log_dict({
            'val/val_loss': loss,
            'val/psnr': psnr,
            'val/ssim': ssim_value
        }, prog_bar=True, sync_dist=True)
        
        # Save validation images periodically
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            # Create validation directory if it doesn't exist
            os.makedirs('validation_images', exist_ok=True)
            
            # Save grid of images (LR, SR, HR)
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
        
        # Store metrics for epoch end averaging
        loss = {
            'val_loss': loss,
            'psnr': psnr,
            'ssim': ssim_value
        }
        
        self.outputs.append(loss)
            
        return loss

    def on_validation_epoch_end(self):
        # Stack all the metrics from validation steps
        avg_gen_loss = torch.stack([x['val_loss'] for x in self.outputs]).mean()
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

# Training script
def train_srcnn():
    model = SRCNN(lr=1e-4)
    datamodule = SRDataModule('mammography_sr_dataset_crop2', batch_size=6)
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/epoch_ssim',
        filename='srcnn-{epoch:02d}-ssim{val/epoch_ssim:.4f}',
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
        logger=pl.loggers.TensorBoardLogger('logs', name='srcnn_runs'),
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
    train_srcnn()