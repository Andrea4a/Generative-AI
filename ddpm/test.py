import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import torch
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
import os
import math
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
IMG_CHANNELS = 3
TIME_ENCODING_SIZE = 64
COND_SHAPE = (3,)
BATCH_SIZE = 64
L = 1000
guidance_scale = 3.0  # w, puoi provare valori tra 1 e 5
n_samples_per_cond = 8

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./generated"
GUIDANCE_SCALE = 3.0
os.makedirs(SAVE_DIR, exist_ok=True)

# === Caricamento modello ===
model = Network().to(device)
checkpoint = torch.load("model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Reimporta le componenti necessarie ===
noise_schedule = NoiseSchedule(L)
time_encoding = TimeEncoding(L, TIME_ENCODING_SIZE)

@torch.no_grad()
def generate_image(cond_vector, guidance_scale=2.0):
    """
    cond_vector: tensor di shape [1, 3] con valori binari (es: torch.tensor([[1, 0, 1]], device=device))
    """
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    for t_inv in reversed(range(L)):
        t = torch.tensor([t_inv], device=device)
        alpha_t = noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
        one_minus_alpha_t = noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)

        # === CFG ===
        eps_cond = model(x, t, cond_vector)
        eps_uncond = model(x, t, torch.zeros_like(cond_vector))
        eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

        # === Aggiornamento step (DDPM) ===
        beta_t = noise_schedule.beta[t].view(-1, 1, 1, 1)
        x = (x - beta_t * eps / one_minus_alpha_t) / torch.sqrt(1.0 - beta_t)

        if t_inv > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise

    return torch.clamp(x, 0, 1)

# Combinazioni condizionate: (Male, Eyeglasses, Beard)
conditions = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
]
labels = [
    "Femmina, No occhiali, Barba",
    "Femmina, No occhiali, No Barba",
    "Femmina, Occhiali, Barba",
    "Femmina, Occhiali, No Barba",
    "Maschio, No occhiali, Barba",
    "Maschio, No occhiali, No Barba",
    "Maschio, Occhiali, Barba",
    "Maschio, Occhiali, No Barba",
]

n_samples_per_cond = 8  # immagini per ogni condizione
SAVE_PATH = "griglia_epoca_154.png"

from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

@torch.no_grad()
def generate_grid():
    all_images = []
    for cond in tqdm(conditions, desc="Generazione griglia"):
        cond_tensor = torch.tensor([cond], dtype=torch.float32, device=device).repeat(n_samples_per_cond, 1)
        x = torch.randn(n_samples_per_cond, 3, IMG_SIZE, IMG_SIZE).to(device)

        for t_inv in reversed(range(L)):
            t = torch.full((n_samples_per_cond,), t_inv, device=device, dtype=torch.long)
            alpha_t = noise_schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
            one_minus_alpha_t = noise_schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)

            eps_cond = model(x, t, cond_tensor)
            eps_uncond = model(x, t, torch.zeros_like(cond_tensor))
            eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

            beta_t = noise_schedule.beta[t].view(-1, 1, 1, 1)
            x = (x - beta_t * eps / one_minus_alpha_t) / torch.sqrt(1.0 - beta_t)

            if t_inv > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        x = x.clamp(0, 1).cpu()
        all_images.append(x)

    # Crea la griglia finale 8x8
    final_tensor = torch.cat(all_images, dim=0)
    grid_img = make_grid(final_tensor, nrow=n_samples_per_cond)
    grid_np = (grid_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Crea la colonna con le etichette
    h, w, _ = grid_np.shape
    label_strip = Image.new("RGB", (150, h), (255, 255, 255))
    draw = ImageDraw.Draw(label_strip)
    font = ImageFont.load_default()

    for i, label in enumerate(labels):
        y = i * IMG_SIZE + IMG_SIZE // 2 - 6
        draw.text((5, y), label, fill=(0, 0, 0), font=font)

    # Combina etichette + griglia
    image = Image.fromarray(grid_np)
    final_img = Image.new("RGB", (150 + w, h))
    final_img.paste(label_strip, (0, 0))
    final_img.paste(image, (150, 0))

    final_img.save(SAVE_PATH)
    print(f"✅ Griglia con etichette salvata in {SAVE_PATH}")

# === Esegui la generazione
generate_grid()
