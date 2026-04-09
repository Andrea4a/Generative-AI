# Generative AI on CelebA: GAN, VAE, and DDPM

This repository collects three generative modeling approaches for conditional face generation on the **CelebA** dataset:

- **Conditional GAN**
- **Conditional VAE**
- **Conditional DDPM**

All models are trained to generate **64×64 RGB face images** conditioned on three binary facial attributes:

- **Gender**
- **Eyeglasses**
- **Beard**

The project is organized into three independent modules, each with its own training and testing scripts or notebooks.

## Overview

The goal of this project is to compare different generative approaches for **attribute-conditioned face synthesis**.

Each model uses the same conditioning scheme based on three CelebA attributes and is designed to generate images consistent with the requested combination of labels.

The project includes:

- a **Conditional GAN** trained with adversarial loss and an additional diversity term
- a **Conditional Variational Autoencoder (CVAE)** for latent-variable-based generation
- a **Conditional DDPM** with classifier-free guidance for diffusion-based image synthesis

---

## Conditioning Attributes

The models are conditioned on the following three CelebA attributes:

- **Male**
- **Eyeglasses**
- **Beard**

In the code, these attributes are selected from CelebA using fixed indices.

The conditioning vector has shape:

```python
[gender, eyeglasses, beard]
```

and each value is binary.

## Dataset

All models are trained on **CelebA**.

The training scripts expect the dataset to be available locally at:

```text
/home/pfoggia/GenerativeAI/CELEBA
```

If your dataset is stored elsewhere, update the dataset path inside the scripts before training.

## How to Run

## GAN

Train the conditional GAN:

```bash
cd gan
python train.py
```

Use the notebook for testing or visual analysis:

```text
gan/test.ipynb
```

---

## VAE

Train the conditional VAE:

```bash
cd vae
python train.py --epochs 20 --batch_size 64 --lr 1e-4 --latent_dim 128
```

Use the notebook for testing and generation:

```text
vae/test.ipynb
```

---

## DDPM

Train the conditional DDPM:

```bash
cd ddpm
python train.py
```

Generate images with a trained model:

```bash
cd ddpm
python test.py
```
