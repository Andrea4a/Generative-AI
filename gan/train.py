import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import itertools
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

# Parametri
batch_size = 128
image_size = 64
z_dim = 100
cond_dim = 3
ngf = 128  
ndf = 128  
nc = 3
diversity_weight = 0.3  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_fixed = torch.randn(8, z_dim, device=device)

# Dataset CelebA + preprocessing
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = torchvision.datasets.CelebA(
    root="/home/pfoggia/GenerativeAI/CELEBA",
    split="train",
    target_type="attr",
    download=False,
    transform=transform
)

attrib_idx = [20, 15, 24]

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack([target[attrib_idx] for target in targets])
    labels = (labels + 1) // 2
    return images, labels

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 + cond_dim, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        batch = z.size(0)
        x = self.fc(z).view(batch, ngf * 8, 4, 4)
        cond_map = labels.view(batch, cond_dim, 1, 1).expand(-1, -1, 4, 4)
        x = torch.cat([x, cond_map], dim=1)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.main = nn.Sequential(
            block(nc + cond_dim, ndf),
            block(ndf, ndf * 2),
            block(ndf * 2, ndf * 4),
            block(ndf * 4, ndf * 8),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        batch = img.size(0)
        cond_map = labels.view(batch, cond_dim, 1, 1).expand(-1, -1, image_size, image_size)
        x = torch.cat([img, cond_map], dim=1)
        return self.main(x).view(-1, 1)

# Loss
def generator_loss(d_synth):
    return F.binary_cross_entropy(d_synth, torch.full_like(d_synth, 0.9))

def discriminator_loss(d_true, d_fake):
    return (F.binary_cross_entropy(d_true, torch.full_like(d_true, 0.9)) +
            F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake)))

def decode_label(label_tensor):
    g, e, b = [int(v) for v in label_tensor]
    gender = 'F' if g == 0 else 'M'
    glasses = 'Y' if e == 1 else 'N'
    beard = 'N' if b == 1 else 'Y'
    return f"{gender}_{glasses}_{beard}"

def save_samples(netG, epoch):
    netG.eval()
    rows = []
    labels_all = []

    with torch.no_grad():
        for label in itertools.product([0., 1.], repeat=3):
            label_tensor = torch.tensor(label, device=device).unsqueeze(0).repeat(8, 1)
            gen_imgs = netG(z_fixed, label_tensor).detach().cpu()
            rows.append(gen_imgs)
            labels_all.append(decode_label(label))

    full_grid = torch.cat(rows, dim=0)
    grid = torchvision.utils.make_grid(full_grid, nrow=8, normalize=True)

    grid_img = TF.to_pil_image(grid)
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    for i, label_str in enumerate(labels_all):
        y = i * (image_size + 2) + 10
        draw.text((2, y), label_str, fill=(255, 255, 255), font=font)

    os.makedirs("samples", exist_ok=True)
    grid_img.save(f"samples/epoch_{epoch}_grid_labeled.png")
    netG.train()

# Istanze
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=4e-4, betas=(0.5, 0.999))

os.makedirs("samples", exist_ok=True)
fixed_labels = torch.tensor(list(itertools.product([0., 1.], repeat=3)), device=device)
fixed_z = torch.randn(len(fixed_labels), z_dim, device=device)

epochs = 200
for epoch in range(epochs):
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch = imgs.size(0)

        # Discriminator
        z = torch.randn(batch, z_dim, device=device)
        gen_imgs = netG(z, labels)
        loss_D = discriminator_loss(netD(imgs, labels), netD(gen_imgs.detach(), labels))
        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()

        # Generator con diversity loss
        z1 = torch.randn(batch, z_dim, device=device)
        z2 = torch.randn(batch, z_dim, device=device)
        gen1 = netG(z1, labels)
        gen2 = netG(z2, labels)

        adv_loss = generator_loss(netD(gen1, labels))
        div_loss = F.l1_loss(gen1, gen2)
        loss_G = adv_loss - diversity_weight * div_loss
        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()

    print(f"[{epoch+1}/{epochs}] D: {loss_D.item():.4f} | G: {loss_G.item():.4f} | Div: {-div_loss.item():.4f}", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(netG.state_dict(), f"checkpoints/gen_epoch_{epoch+1}.pth")
    save_samples(netG, epoch+1)
