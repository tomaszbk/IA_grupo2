import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleCNN(nn.Module):
    def __init__(self, input_size=(3, 128, 128)):
        super(BottleCNN, self).__init__()

        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calcular automáticamente el tamaño de entrada para la capa fully connected
        self.flatten_dim = self._get_flatten_dim(input_size)

        # Capas totalmente conectadas
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_flatten_dim(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten dinámico
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
