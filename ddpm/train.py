import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
import os
import math
import time
from tqdm import tqdm

# === Dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Parametri ===
IMG_SIZE = 64
IMG_CHANNELS = 3
TIME_ENCODING_SIZE = 64
COND_SHAPE = (3,)
BATCH_SIZE = 64
L = 1000
SAVE_DIR = "checkpoints"
SAVE_INTERVAL_MIN = 10
SAVE_EPOCH_INTERVAL = 50
DROPOUT_PROB = 0.1  # probabilità di dropout per classifier-free guidance

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# === Dataset ===
dataset = CelebA(root="/home/pfoggia/GenerativeAI/CELEBA", split="train",
                 target_type="attr", transform=transform, download=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# === Noise Schedule ===
class NoiseSchedule:
    def __init__(self, L, beta_min=1e-4, beta_max=0.02):
        self.L = L
        self.beta = torch.linspace(beta_min, beta_max, L).to(device)
        self.alpha = torch.cumprod(1.0 - self.beta, dim=0)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_1_alpha = torch.sqrt(1 - self.alpha)

noise_schedule = NoiseSchedule(L)

# === Time Encoding ===
class TimeEncoding:
    def __init__(self, L, dim):
        dim2 = dim // 2
        encoding = torch.zeros(L, dim)
        ang = torch.linspace(0.0, torch.pi / 2, L)
        logmul = torch.linspace(0.0, math.log(40), dim2)
        mul = torch.exp(logmul)
        for i in range(dim2):
            a = ang * mul[i]
            encoding[:, 2 * i] = torch.sin(a)
            encoding[:, 2 * i + 1] = torch.cos(a)
        self.encoding = encoding.to(device)

    def __getitem__(self, t):
        return self.encoding[t]

time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

# === UNet Condizionato ===
class UNetBlock(nn.Module):
    def __init__(self, size, outer_features, inner_features, cond_features, inner_block=None):
        super().__init__()
        self.size = size
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.cond_features = cond_features
        self.encoder = self.build_encoder(outer_features + cond_features, inner_features)
        self.decoder = self.build_decoder(inner_features + cond_features + TIME_ENCODING_SIZE, outer_features)
        self.combiner = self.build_combiner(2 * outer_features, outer_features)
        self.inner = inner_block

    def forward(self, x, time_encodings, cond):
        x0 = x
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, self.size, self.size)
        x = torch.cat((x, cc), dim=1)
        y = self.encoder(x)
        if self.inner:
            y = self.inner(y, time_encodings, cond)
        half_size = self.size // 2
        cc = cond.view(-1, self.cond_features, 1, 1).expand(-1, -1, half_size, half_size)
        tt = time_encodings.view(-1, TIME_ENCODING_SIZE, 1, 1).expand(-1, -1, half_size, half_size)
        y1 = torch.cat((y, cc, tt), dim=1)
        x1 = self.decoder(y1)
        x2 = torch.cat((x1, x0), dim=1)
        return self.combiner(x2)

    def build_combiner(self, from_features, to_features):
        return nn.Conv2d(from_features, to_features, 1)

    def build_encoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.Conv2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )

    def build_decoder(self, from_features, to_features):
        return nn.Sequential(
            nn.Conv2d(from_features, from_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(from_features),
            nn.ReLU(),
            nn.ConvTranspose2d(from_features, to_features, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(to_features),
            nn.ReLU()
        )

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU())
        self.unet = self.build_unet(64, [128, 256, 512, 1024])
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1))  # output a 3 canali (RGB)

    def forward(self, x, t, cond):
        enc = time_encoding[t]
        x = self.pre(x)
        y = self.unet(x, enc, cond)
        y = self.post(y)
        return y

    def build_unet(self, size, feat_list):
        if len(feat_list) > 2:
            inner_block = self.build_unet(size // 2, feat_list[1:])
        else:
            inner_block = None
        return UNetBlock(size, feat_list[0], feat_list[1], COND_SHAPE[0], inner_block)

model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
os.makedirs(SAVE_DIR, exist_ok=True)

# === Caricamento Checkpoint se esistente ===
checkpoint_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.pth')])

if checkpoint_files:
    last_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(SAVE_DIR, last_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"✅ Checkpoint caricato da {checkpoint_path}, partenza da epoca {start_epoch + 1}.")
else:
    print("⚠️ Nessun checkpoint trovato, inizio da zero.")
    start_epoch = 0

# === Training con Classifier-Free Guidance ===
print("Inizio training...")
last_save_time = time.time()

for epoch in range(start_epoch, 2000):
    model.train()
    total_loss = 0
    print(f"\nInizio epoca {epoch+1}...")
    for batch_idx, (x, attr) in enumerate(tqdm(dataloader, desc=f"Epoca {epoch+1}", leave=False)):
        x = x.to(device)
        cond = ((attr[:, [20, 15, 24]] + 1) // 2).float().to(device)

        # Classifier-free guidance: drop condition per una parte del batch
        drop_mask = (torch.rand(x.size(0)) < DROPOUT_PROB).to(device)
        cond[drop_mask] = 0

        t = torch.randint(0, L, (x.size(0),), device=device).long()
        noise = torch.randn_like(x)
        alpha = noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        one_minus_alpha = noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)
        zt = alpha * x + one_minus_alpha * noise
        pred = model(zt, t, cond)
        loss = loss_fn(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if time.time() - last_save_time >= SAVE_INTERVAL_MIN * 60:
            temp_ckpt_name = os.path.join(SAVE_DIR, f"checkpoint_epoca_{epoch+1:03d}_batch_{batch_idx+1:04d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, temp_ckpt_name)
            print(f"Checkpoint temporaneo salvato: {temp_ckpt_name}")
            last_save_time = time.time()

    if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
        epoch_ckpt_name = os.path.join(SAVE_DIR, f"checkpoint_epoca_{epoch+1:03d}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, epoch_ckpt_name)
        print(f"Checkpoint epoca {epoch+1} salvato: {epoch_ckpt_name}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoca {epoch+1} completata. Loss media = {avg_loss:.4f}")

print("Training completato con successo!")
