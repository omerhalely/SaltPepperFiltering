import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_Encoder(nn.Module):
    def __init__(self, n_channels, bilinear):
        super(UNet_Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNet_Deocder(nn.Module):
    def __init__(self, bilinear, n_classes):
        super(UNet_Deocder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = UNet_Encoder(n_channels=n_channels, bilinear=bilinear)
        self.middle_conv = nn.Conv2d(1536, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder = UNet_Deocder(bilinear=bilinear, n_classes=n_classes)

    def forward(self, rgb, x):
        x1, x2, x3, x4, x5 = self.encoder(rgb)
        x5 = torch.cat((x5, x), dim=1)
        x5 = F.relu(self.middle_conv(x5))
        x = self.decoder(x1, x2, x3, x4, x5)
        return x


class doubleconv_3d(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(doubleconv_3d, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(),
            nn.Conv3d(output_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.doubleconv(x)


class down_3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super(down_3d, self).__init__()
        self.conv = doubleconv_3d(input_channels=input_channels, output_channels=output_channels)
        self.down = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x


class rgb_encoder_3d(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(rgb_encoder_3d, self).__init__()
        self.down1 = down_3d(input_channels=input_channels, output_channels=32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down2 = down_3d(input_channels=32, output_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down3 = down_3d(input_channels=64, output_channels=128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down4 = down_3d(input_channels=128, output_channels=output_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class imu_UpSample(nn.Module):
    def __init__(self, imu_length, lstm_output_channels, lstm_layers):
        super(imu_UpSample, self).__init__()
        self.imu_lstm = nn.LSTM(input_size=imu_length, hidden_size=lstm_output_channels, num_layers=lstm_layers,
                                batch_first=True)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(lstm_output_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=2)
        x, _ = self.imu_lstm(x)
        x = torch.unsqueeze(x, dim=2)
        x = torch.permute(x, (0, 3, 2, 1))
        x = self.upsample(x)
        return x


class Depth_3d_model(nn.Module):
    def __init__(self, imu_length, encoder_output_channels, lstm_layers, sequence_length):
        super(Depth_3d_model, self).__init__()
        self.sequence_length = sequence_length
        self.imu_upsample = imu_UpSample(imu_length=imu_length, lstm_output_channels=encoder_output_channels // 16,
                                         lstm_layers=lstm_layers)

        self.rgb_encoder = rgb_encoder_3d(input_channels=3, output_channels=encoder_output_channels)

        self.n_channels = 6
        self.n_classes = 1
        self.bilinear = True

        self.model = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=False)

    def forward(self, rgb, imu):
        imu = self.imu_upsample(imu)

        rgb = torch.reshape(rgb, (1, int(rgb.size(1) / self.sequence_length), self.sequence_length, rgb.size(2), rgb.size(3)))
        encoded_rgb = self.rgb_encoder(rgb)
        encoded_rgb = torch.squeeze(encoded_rgb, dim=0)
        encoded_rgb = torch.permute(encoded_rgb, (1, 0, 2, 3))

        imu = F.interpolate(imu, (encoded_rgb.size(2), encoded_rgb.size(3)))

        imu_rgb = torch.cat((encoded_rgb, imu), dim=1)

        rgb = torch.reshape(rgb, (rgb.size(0), rgb.size(1) * rgb.size(2), rgb.size(3), rgb.size(4)))
        output = self.model(rgb, imu_rgb)

        return output


if __name__ == "__main__":
    height = 240
    width = 320
    imu_length = 10
    encoder_output_channels = 256
    lstm_layers = 3
    sequence_length = 2
    model = Depth_3d_model(imu_length=imu_length,
                       encoder_output_channels=encoder_output_channels,
                       lstm_layers=lstm_layers,
                       sequence_length=sequence_length)
    # torch.save(model.state_dict(), "./new_model.pt")
    rgb = torch.rand(1, 3 * sequence_length, height, width)
    imu = torch.rand(1, 2, 1, 10)

    start = time.time()
    print(model(rgb, imu).shape)
    end = time.time()
    print(end - start)
