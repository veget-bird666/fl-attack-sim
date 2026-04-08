import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleDeepFM(nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SimpleDeepFM, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.dnn = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        fm_out = self.linear(x)
        dnn_out = self.dnn(x)
        return fm_out + dnn_out

def load_mock_adult_data(num_samples=1000):
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)