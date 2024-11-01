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
    def __init__(self, lr: float = 1e-4, target_size: Tuple[int, int] = (1760, 1400)):
        super().__init__()
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

    def validation_step(self, batch, batch_idx):
        low_res, high_res = batch
        sr_image = self(low_res)
        loss = self.loss(sr_image, high_res)
        self.log('val_loss', loss)

        # Save example images every 10 epochs for the first batch
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            os.makedirs('validation_examples', exist_ok=True)
            
            # Save original low-res, high-res, and super-resolved images
            save_image(
                low_res, 
                f'validation_examples/epoch_{self.current_epoch+1}_low_res.png'
            )
            save_image(
                high_res, 
                f'validation_examples/epoch_{self.current_epoch+1}_high_res.png'
            )
            save_image(
                sr_image, 
                f'validation_examples/epoch_{self.current_epoch+1}_super_res.png'
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
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
    datamodule = SRDataModule('mammography_sr_dataset', batch_size=8)
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='srcnn-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_srcnn()