"""Deep Learning Convolutional Neural Network (CNN) for Automatic Modulation Classification (AMC).

Operates on raw baseband (I, Q) tensor representations: shape (Batch, 2, Samples).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationCNN(nn.Module):
    """1D Convolutional Neural Network for RF modulation recognition.

    Inputs:
        Tensor of shape (batch_size, 2, num_samples), where dim 1 corresponds to [I, Q].
    Outputs:
        Logits of shape (batch_size, num_classes).
    """

    CLASSES: List[str] = ["AM", "FM", "BPSK", "QPSK", "16QAM"]

    def __init__(self, num_classes: int = 5, input_samples: int = 1024):
        super().__init__()
        self.num_classes = num_classes
        self.input_samples = input_samples

        # Feature Extractor: 1D Convolutions over (I, Q) channels
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.4)

        # Calculate flattened dimension after 3 pooling layers
        reduced_len = input_samples // 8
        self.fc1 = nn.Linear(128 * reduced_len, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Dense classifier
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.out(x)
        return logits

    @classmethod
    def iq_to_tensor(cls, iq_samples: "np.ndarray", target_len: int = 1024) -> torch.Tensor:
        """Convert complex numpy array of IQ samples into normalized PyTorch tensor (1, 2, target_len)."""
        import numpy as np

        if len(iq_samples) < target_len:
            iq_samples = np.pad(iq_samples, (0, target_len - len(iq_samples)))
        else:
            iq_samples = iq_samples[:target_len]

        # Power normalization
        pwr = np.sqrt(np.mean(np.abs(iq_samples) ** 2)) + 1e-12
        normalized = iq_samples / pwr

        i_channel = np.real(normalized).astype(np.float32)
        q_channel = np.imag(normalized).astype(np.float32)
        tensor = torch.tensor(np.stack([i_channel, q_channel]), dtype=torch.float32)
        return tensor.unsqueeze(0)  # Shape (1, 2, target_len)
