import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import time
import os

# Parsing argomenti
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--save_path', type=str, default='vae_cond_celeba.pt')
args = parser.parse_args()

# Dataset CelebA
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor()
])

celeba = datasets.CelebA(root='/home/pfoggia/GenerativeAI/CELEBA',
                         split='train', target_type='attr',
                         transform=transform, download=False)

dataloader = DataLoader(celeba, batch_size=args.batch_size, shuffle=True)

# Attributi da usare (Gender, Eyeglasses, Beard)
attribute_indices = [20, 15, 24]

# Definizione del modello CVAE
class CVAE(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + cond_dim, 64, 4, 2, 1),     # 64x64 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),              # 32x32 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),             # 16x16 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),             # 8x8 → 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten()
        )
        self.encoder_out_dim = 512 * 4 * 4
        self.linear_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.linear_log_sigma = nn.Linear(self.encoder_out_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + cond_dim, self.encoder_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),    # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # 16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),      # 32x32 → 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 1, 1),       # 64x64 → 64x64
            nn.Sigmoid()
        )

    def forward(self, x, cond):
    # Espansione condizione come mappe e concatenazione all'immagine
     B, _, H, W = x.shape
     male, glasses, no_beard = cond[:, 0], cond[:, 1], cond[:, 2]
     male_tensor = male.view(-1, 1, 1, 1).expand(-1, 1, H, W).to(x.device)
     glasses_tensor = glasses.view(-1, 1, 1, 1).expand(-1, 1, H, W).to(x.device)
     no_beard_tensor = no_beard.view(-1, 1, 1, 1).expand(-1, 1, H, W).to(x.device)
     cond_maps = torch.cat([male_tensor, glasses_tensor, no_beard_tensor], dim=1)
     x_cond = torch.cat([x, cond_maps], dim=1)
    
    # Passaggio nell'encoder
     out = self.encoder(x_cond)
     

     mu = self.linear_mu(out)
     log_sigma = self.linear_log_sigma(out)

 
     eps = torch.randn_like(log_sigma)
     z = mu + eps * torch.exp(log_sigma)
     
    # Decoder con condizione concatenata
     z_cond = torch.cat([z, cond], dim=1)
     x = self.decoder_input(z_cond).view(-1, 512, 4, 4)
     

     y = self.decoder(x)
     

     y = torch.clamp(y, 0.0, 1.0)
     return y, mu, log_sigma


    # Funzione per estrarre vettori latenti z
    def compute_latent_vectors(self, x):
        out = self.encoder(x)
        mu = self.linear_mu(out)
        log_sigma = self.linear_log_sigma(out)
        eps = torch.randn_like(log_sigma)
        z = mu + eps * torch.exp(log_sigma)
        return z

# Loss Function
reconstruction_loss_function=nn.BCELoss(reduction='sum')
def kl_loss_function(mu, log_sigma):
    kl=0.5*(mu**2 + torch.exp(2*log_sigma)-1-2*log_sigma)
    return torch.sum(kl)

beta=1.0  # Peso della loss KL
def vae_loss(recon_x, x, mu, log_sigma):
    recon_loss = reconstruction_loss_function(recon_x, x)
    return recon_loss + beta * kl_loss_function(mu, log_sigma)



# Setup training
device_str = os.getenv('TORCH_DEVICE', 'cpu')
if not device_str:
    device_str = 'cpu'
device = torch.device(device_str)

print(f"[INFO] Using device: {device}", flush=True)
model = CVAE(args.latent_dim, cond_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
checkpoint_dir = "checkpoints3"
os.makedirs(checkpoint_dir, exist_ok=True)
latest_ckpt = None
start_epoch = 0

if os.path.exists(checkpoint_dir):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_epoch")]
    if ckpts:
        #Riprende dall'ultimo checkpoint
        latest_ckpt = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))[-1]
        ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Ripristinato da checkpoint: {latest_ckpt}, riparto da epoch {start_epoch}", flush=True)
    else:
        print("[INFO] Nessun checkpoint trovato, inizio da capo", flush=True)

# Training loop
start_time = time.time()
for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0.0

    for imgs, attrs in dataloader:
        imgs = imgs.to(device)
        cond = attrs[:, attribute_indices].float().to(device)
        cond = (cond + 1) / 2  # normalizzazione attributi

        optimizer.zero_grad()
        recon, mu, log_sigma = model(imgs, cond)
        loss = vae_loss(recon, imgs, mu, log_sigma)
        loss.backward()
        optimizer.step()

        # EMA loss (media mobile esponenziale)
        average_loss = 0.9 * average_loss + 0.1 * loss.item() if 'average_loss' in locals() else loss.item()

        

        total_loss += loss.item()

        # Salvataggio checkpoint ogni 10 minuti
        if time.time() - start_time > 600:
            ckpt_path = f"checkpoints3/ckpt_epoch{epoch+1}_step{int(time.time())}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, ckpt_path)
            print(f"\n[Checkpoint salvato] {ckpt_path}", flush=True)
            start_time = time.time()

    avg_epoch_loss = total_loss / len(dataloader.dataset)
    print(f"\nEpoch {epoch+1}/{args.epochs}, Average Loss: {average_loss:.4f}", flush=True)

# Salvataggio modello

torch.save(model.state_dict(), args.save_path)
print("Modello salvato:", args.save_path, flush=True)
