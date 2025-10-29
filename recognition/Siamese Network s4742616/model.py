import torch
import torch.nn as nn
import torchvision

INPUT_CHANNELS = 3
IMAGE_SIZE = 256


class FeatureExtractor(nn.module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2)
        )
        self.connected = nn.sequential(
            nn.Linear(INPUT_CHANNELS*IMAGE_SIZE**2*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256)

        )
    def forward(self, x):
        x = self.cnn(x) #Apply Convolution
        x = x.view(x.size(0), -1) #Flatten to 1D
        x = self.connected(x)
        return x
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()

    def forward(self, x1, x2):
        image1 = self.feature_extractor(x1)
        image2 = self.feature_extractor(x2)

        distance = torch.abs(image1 - image2)

        return distance