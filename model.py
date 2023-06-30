import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels):
        self.input_channels = input_channels
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(input_channels, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

class salt_pepper_filtering_model(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder = Decoder(input_channels=64)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    input_channels = 1
    batch_size = 10
    height = 265
    width = 315
    model = salt_pepper_filtering_model(input_channels=input_channels)

    image = torch.rand(batch_size, input_channels, height, width)

    print(model(image).shape)
