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
            
            # Third 2x upscale
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
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
        self.automatic_optimization = False  # Disable automatic optimization
        
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lr = lr
        
        # Loss functions
        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        
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

    def forward(self, x):
        return self.generator(x)

    def adversarial_criterion(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.adversarial_loss(pred, target)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
                   
        # Train Generator
        opt_g.zero_grad()
        
        # Generate SR images
        sr_imgs = self(lr_imgs)
            
        gen_validity = self.discriminator(sr_imgs)
        
        # Calculate generator losses
        content_loss = self.content_loss(sr_imgs, hr_imgs)
        perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
        adversarial_loss = self.adversarial_criterion(gen_validity, True)
        
        gen_loss = (self.content_weight * content_loss + 
                   self.perceptual_weight * perceptual_loss +
                   self.adversarial_weight * adversarial_loss)
        
        self.manual_backward(gen_loss)
        opt_g.step()
        
        # Train Discriminator
        opt_d.zero_grad()
        
        # Real loss
        real_validity = self.discriminator(hr_imgs)
        real_loss = self.adversarial_criterion(real_validity, True)
        
        # Fake loss
        fake_validity = self.discriminator(sr_imgs.detach())
        fake_loss = self.adversarial_criterion(fake_validity, False)
        
        d_loss = (real_loss + fake_loss) / 2
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Log losses
        self.log_dict({
            'g_loss': gen_loss,
            'content_loss': content_loss,
            'perceptual_loss': perceptual_loss,
            'adversarial_loss': adversarial_loss,
            'd_loss': d_loss
        })

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
                    
        sr_imgs = self(lr_imgs)
        
        # print(f"LR shape: {lr_imgs.shape}")
        # print(f"HR shape: {hr_imgs.shape}")
        # print(f"SR shape: {sr_imgs.shape}")

        content_loss = self.content_loss(sr_imgs, hr_imgs)
        self.log('val_loss', content_loss)
        
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            os.makedirs('validation_examples', exist_ok=True)
            save_image(lr_imgs, f'validation_examples/epoch_{self.current_epoch+1}_low_res.png')
            save_image(hr_imgs, f'validation_examples/epoch_{self.current_epoch+1}_high_res.png')
            save_image(sr_imgs, f'validation_examples/epoch_{self.current_epoch+1}_super_res.png')

        return content_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        return [opt_g, opt_d]

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
    datamodule = SRDataModule('mammography_sr_dataset', batch_size=2)
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='srgan-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min'
            ),
            # pl.callbacks.EarlyStopping(
            #     monitor='val_loss',
            #     patience=10,
            #     mode='min'
            # )
        ]
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_srgan()