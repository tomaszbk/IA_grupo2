import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

class BottleCNN(pl.LightningModule):
    def __init__(self, input_size=(1, 128, 128), learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.calculate_out_dim(input_size)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.loss_fn = nn.BCELoss()
        self.val_accuracy = Accuracy(task="binary")

    def calculate_out_dim(self, input_size):
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size)  # (batch, C, H, W)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            self._to_linear = x.view(1, -1).size(1)  # Tamaño calculado

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)  # Squeeze the last dimension to match target shape

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        self.train_acc(y_hat, y)  # Calcula accuracy
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)  # Loggea accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        self.test_acc(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # Usa self.learning_rate