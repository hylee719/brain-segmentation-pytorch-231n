# from collections import OrderedDict

# import torch
# import torch.nn as nn


# class UNet(nn.Module):

#     def __init__(self, in_channels=3, out_channels=1, init_features=32):
#         super(UNet, self).__init__()

#         features = init_features
#         self.encoder1 = UNet._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = UNet._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#     self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

#     self.upconv4 = nn.ConvTranspose2d(
#         features * 16, features * 8, kernel_size=2, stride=2
#     )
#     self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
#     self.upconv3 = nn.ConvTranspose2d(
#         features * 8, features * 4, kernel_size=2, stride=2
#     )
#     self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
#     self.upconv2 = nn.ConvTranspose2d(
#         features * 4, features * 2, kernel_size=2, stride=2
#     )
#     self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
#     self.upconv1 = nn.ConvTranspose2d(
#         features * 2, features, kernel_size=2, stride=2
#     )
#     self.decoder1 = UNet._block(features * 2, features, name="dec1")

#     self.conv = nn.Conv2d(
#         in_channels=features, out_channels=out_channels, kernel_size=1
#     )

# def forward(self, x):
#     enc1 = self.encoder1(x)
#     enc2 = self.encoder2(self.pool1(enc1))
#     enc3 = self.encoder3(self.pool2(enc2))
#     enc4 = self.encoder4(self.pool3(enc3))

#     bottleneck = self.bottleneck(self.pool4(enc4))

#     dec4 = self.upconv4(bottleneck)
#     dec4 = torch.cat((dec4, enc4), dim=1)
#     dec4 = self.decoder4(dec4)
#     dec3 = self.upconv3(dec4)
#     dec3 = torch.cat((dec3, enc3), dim=1)
#     dec3 = self.decoder3(dec3)
#     dec2 = self.upconv2(dec3)
#     dec2 = torch.cat((dec2, enc2), dim=1)
#     dec2 = self.decoder2(dec2)
#     dec1 = self.upconv1(dec2)
#     dec1 = torch.cat((dec1, enc1), dim=1)
#     dec1 = self.decoder1(dec1)
#     return torch.sigmoid(self.conv(dec1))

# @staticmethod
# def _block(in_channels, features, name):
#     return nn.Sequential(
#         OrderedDict(
#             [
#                 (
#                     name + "conv1",
#                     nn.Conv2d(
#                         in_channels=in_channels,
#                         out_channels=features,
#                         kernel_size=3,
#                         padding=1,
#                         bias=False,
#                     ),
#                 ),
#                 (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                 (name + "relu1", nn.ReLU(inplace=True)),
#                 (
#                     name + "conv2",
#                     nn.Conv2d(
#                         in_channels=features,
#                         out_channels=features,
#                         kernel_size=3,
#                         padding=1,
#                         bias=False,
#                     ),
#                 ),
#                 (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                 (name + "relu2", nn.ReLU(inplace=True)),
#             ]
#         )
#     )

from collections import OrderedDict
import torch
import torch.nn as nn
from vit_pytorch import ViT


class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super(ViTEncoder, self).__init__()
        # self.vit = ViT(
        #     image_size=256,
        #     patch_size=16,
        #     num_classes=1000,
        #     dim=768,
        #     depth=12,
        #     heads=12,
        #     mlp_dim=3072,
        #     pool="cls",
        #     channels=in_channels,
        # )
        # self.linear = nn.Linear(768, features)
        self.vit = ViT(
            image_size=256,
            patch_size=16,
            num_classes=1,  # Set to 1 for binary segmentation (out_channels=1)
            dim=32,  # Use the same value as the init_features in the UNet
            depth=8,  # Set based on the number of encoder and decoder blocks in the UNet
            heads=12,  # Experiment with different values to find the optimal number of heads
            mlp_dim=128,  # Use the same value as the init_features in the UNet
            pool="cls",
            channels=3,  # Set to the number of input channels in the UNet
        )
        self.linear = nn.Linear(32, features)

    def forward(self, x):
        x = self.vit(x)
        x = self.linear(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = ViTEncoder(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Rest of the UNet architecture remains the same
        self.bottleneck = UNet._block(
            features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
