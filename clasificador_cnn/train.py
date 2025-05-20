import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from cnn_model import BottleCNN
import os

# Configuración general
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 128
DATA_DIR = "../data/train"
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Dataset
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo

model = BottleCNN().to(DEVICE)
trainer = Trainer(max_epochs=EPOCHS, gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(model, train_loader)
