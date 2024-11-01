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
        
        low_res = Image.open(os.path.join(self.low_res_dir, img_name)).convert('L')
        high_res = Image.open(os.path.join(self.high_res_dir, img_name)).convert('L')
        
        low_res = self.transform(low_res)
        high_res = self.transform(high_res)
        
        return low_res, high_res

class DenseBlock(nn.Module):
    def __init__(self, channels, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.beta = 0.2

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.beta + x

class RRDB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dense1 = DenseBlock(channels)
        self.dense2 = DenseBlock(channels)
        self.dense3 = DenseBlock(channels)
        self.beta = 0.2

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * self.beta + x

class RRDBNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=23, scale_factor=8):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # RRDB blocks
        self.body = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Upsampling blocks (8x = 2^3)
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        out = self.upsampling(feat)
        out = self.conv_last(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.features = nn.Sequential(
            *discriminator_block(1, 64, 2, False),
            *discriminator_block(64, 128, 2),
            *discriminator_block(128, 256, 2),
            *discriminator_block(256, 512, 2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

class ESRGAN(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.automatic_optimization = False
        
        self.generator = RRDBNet()
        self.discriminator = Discriminator()
        self.lr = lr
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = monai.losses.PerceptualLoss(
            spatial_dims=2,
            network_type="squeeze",
            is_fake_3d=False,
            pretrained=True
        )
        
        # Loss weights
        self.pixel_weight = 1
        self.perceptual_weight = 1
        self.gan_weight = 0.1

    def forward(self, x):
        return self.generator(x)

    def compute_gan_loss(self, real_pred, fake_pred):
        # Relativistic average GAN loss
        real_avg = real_pred - fake_pred.mean()
        fake_avg = fake_pred - real_pred.mean()
        
        loss_real = F.binary_cross_entropy_with_logits(
            real_avg, torch.ones_like(real_avg)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_avg, torch.zeros_like(fake_avg)
        )
        
        return (loss_real + loss_fake) / 2

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        lr_imgs, hr_imgs = batch
        
        # Train Generator
        opt_g.zero_grad()
        
        sr_imgs = self(lr_imgs)
        real_pred = self.discriminator(hr_imgs)
        fake_pred = self.discriminator(sr_imgs)
        
        # Calculate losses
        pixel_loss = self.l1_loss(sr_imgs, hr_imgs)
        perceptual_loss = self.perceptual_loss(sr_imgs, hr_imgs)
        gan_loss = self.compute_gan_loss(real_pred, fake_pred)
        
        gen_loss = (self.pixel_weight * pixel_loss + 
                   self.perceptual_weight * perceptual_loss +
                   self.gan_weight * gan_loss)
        
        self.manual_backward(gen_loss)
        opt_g.step()
        
        # Train Discriminator
        opt_d.zero_grad()
        
        real_pred = self.discriminator(hr_imgs)
        fake_pred = self.discriminator(sr_imgs.detach())
        
        d_loss = self.compute_gan_loss(real_pred, fake_pred)
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Log losses
        self.log_dict({
            'g_loss': gen_loss,
            'pixel_loss': pixel_loss,
            'perceptual_loss': perceptual_loss,
            'gan_loss': gan_loss,
            'd_loss': d_loss
        })

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        sr_imgs = self(lr_imgs)
        
        val_loss = self.l1_loss(sr_imgs, hr_imgs)
        self.log('val_loss', val_loss)
        
        if batch_idx == 0 and (self.current_epoch + 1) % 10 == 0:
            os.makedirs('validation_examples', exist_ok=True)
            save_image(lr_imgs, f'validation_examples/epoch_{self.current_epoch+1}_low_res.png')
            save_image(hr_imgs, f'validation_examples/epoch_{self.current_epoch+1}_high_res.png')
            save_image(sr_imgs, f'validation_examples/epoch_{self.current_epoch+1}_super_res.png')
            
        return val_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.9, 0.99))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.9, 0.99))
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

def train_esrgan():
    model = ESRGAN(lr=1e-4)
    datamodule = SRDataModule('mammography_sr_dataset', batch_size=1)
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='auto',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='esrgan-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min'
            )
        ]
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train_esrgan()