#from collections import OrderedDict
#
#import torch
#import torch.nn as nn
#
#
#class UNet(nn.Module):
#
#    def __init__(self, in_channels=3, out_channels=1, init_features=32):
#        super(UNet, self).__init__()
#
#        features = init_features
#        self.encoder1 = UNet._block(in_channels, features, name="enc1")
#        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.encoder2 = UNet._block(features, features * 2, name="enc2")
#        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
#
#        self.upconv4 = nn.ConvTranspose2d(
#            features * 16, features * 8, kernel_size=2, stride=2
#        )
#        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
#        self.upconv3 = nn.ConvTranspose2d(
#            features * 8, features * 4, kernel_size=2, stride=2
#        )
#        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
#        self.upconv2 = nn.ConvTranspose2d(
#            features * 4, features * 2, kernel_size=2, stride=2
#        )
#        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
#        self.upconv1 = nn.ConvTranspose2d(
#            features * 2, features, kernel_size=2, stride=2
#        )
#        self.decoder1 = UNet._block(features * 2, features, name="dec1")
#
#        self.conv = nn.Conv2d(
#            in_channels=features, out_channels=out_channels, kernel_size=1
#        )
#
#    def forward(self, x):
#       # print('shape before:', x.shape)
#        enc1 = self.encoder1(x)
#        #print('shape after enc1, before enc2:', enc1.shape)
#        pool1 = self.pool1(enc1)
#        #print('shape after enc1 pool1 before enc2:', pool1.shape)
#        enc2 = self.encoder2(pool1)
#        #print('shape after enc2:', enc2.shape)
#        enc3 = self.encoder3(self.pool2(enc2))
#        enc4 = self.encoder4(self.pool3(enc3))
#
#        bottleneck = self.bottleneck(self.pool4(enc4))
#        print('shape after bottleneck (needed):', bottleneck.shape)
#        dec4 = self.upconv4(bottleneck)
#        print('shape after dec4:', dec4.shape)
#        dec4 = torch.cat((dec4, enc4), dim=1)
#        dec4 = self.decoder4(dec4)
#        dec3 = self.upconv3(dec4)
#        dec3 = torch.cat((dec3, enc3), dim=1)
#        dec3 = self.decoder3(dec3)
#        dec2 = self.upconv2(dec3)
#        dec2 = torch.cat((dec2, enc2), dim=1)
#        dec2 = self.decoder2(dec2)
#        dec1 = self.upconv1(dec2)
#        dec1 = torch.cat((dec1, enc1), dim=1)
#        dec1 = self.decoder1(dec1)
#        print("shape after dec1 (Final):", dec1.shape)
#        return torch.sigmoid(self.conv(dec1))
#
#    @staticmethod
#    def _block(in_channels, features, name):
#        return nn.Sequential(
#            OrderedDict(
#                [
#                    (
#                        name + "conv1",
#                        nn.Conv2d(
#                            in_channels=in_channels,
#                            out_channels=features,
#                            kernel_size=3,
#                            padding=1,
#                            bias=False,
#                        ),
#                    ),
#                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                    (name + "relu1", nn.ReLU(inplace=True)),
#                    (
#                        name + "conv2",
#                        nn.Conv2d(
#                            in_channels=features,
#                            out_channels=features,
#                            kernel_size=3,
#                            padding=1,
#                            bias=False,
#                        ),
#                    ),
#                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                    (name + "relu2", nn.ReLU(inplace=True)),
#                ]
#            )
#        )

from collections import OrderedDict
import torch.nn as nn
import torch
from transformers import ViTConfig, ViTModel
from vit_pytorch import ViT
#
#
#class ViTEncoder(nn.Module):
#    def __init__(self, in_channels=3, features=32):
#        super(ViTEncoder, self).__init__()
#        self.features = features
#        self.vit = ViT(
#            image_size=256,
#            patch_size=16,
#            num_classes=256,
#            dim=features,
#            depth=12,
#            heads=12,
#            mlp_dim=features*4,
#            pool="cls",
#            channels=in_channels,
#        )
#        #self.linear = nn.Linear(features * 256 * 256, features * 256 * 256)
#
#    def forward(self, x):
#        x = self.vit(x)
#        #print("SHAPE after VIT forward:", x.shape)
#        #x = self.linear(x)
#        #print("Shape after VIT linear layer:", x.shape)
#        #x = torch.reshape(x, (16, self.features, 256, 256))
#        x = torch.reshape(x, (-1, self.features, 256, 256))
#        #print("Shape after VIT reshape:", x.shape)
#        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        #self.encoder1 = ViTEncoder(in_channels, features)
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
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
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        # ENCODER CONFIGS
        config.image_size = 256
        config.num_attention_heads = 16
        config.hidden_size = 768
        config.num_channels = 3
        config.patch_size = 16

        # BOTTLENECK CONFIGS
        #config.image_size = 16
        #config.num_attention_heads = 16
        #config.hidden_size = 512
        #config.patch_size = 1
        #config.num_channels = 512

        # DECODER CONFIGS
        #config.image_size = 128
        #config.patch_size = 2
        #config.num_attention_heads = 16
        #config.num_channels = 64
        #config.hidden_size = 256
        self.vit = ViTModel(config, add_pooling_layer=False)

    def forward(self, x):
       # print("Starting shape:", x.shape)
       # enc1 = self.encoder1(x)
       # test encoder vit
        enc1 = self.vit(x)
        enc1 = enc1.last_hidden_state[:, 1:].view(-1, 3, 256, 256)
        enc1 = self.encoder1(enc1)
       # print("SHAPE AFTER ENC1:", enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
       # print("SHAPE AFTER ENC2:", enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
       # print("SHAPE AFTER ENC3:", enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))
       # print("SHAPE AFTER ENC4:", enc4.shape)

        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # vit for bottleneck
        #bottleneck = self.vit(bottleneck)
        #bottleneck = bottleneck.last_hidden_state[:, 1:].view(-1, 512, 16, 16)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #dec2 = self.vit(dec2)
        #print("shape after dec2 vit:", dec2.shape)
        #dec2 = dec2.last_hidden_state[:, 1:].view(-1, 64, 128, 128)
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
