# from collections import OrderedDict

# import torch
# import torch.nn as nn


# class UNet(nn.Module):

#    def __init__(self, in_channels=3, out_channels=1, init_features=32):
#        super(UNet, self).__init__()

#        features = init_features
#        self.encoder1 = UNet._block(in_channels, features, name="enc1")
#        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.encoder2 = UNet._block(features, features * 2, name="enc2")
#        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

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

#        self.conv = nn.Conv2d(
#            in_channels=features, out_channels=out_channels, kernel_size=1
#        )

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

# from collections import OrderedDict
# import torch.nn as nn
# import torch
# from vit_pytorch import ViT
# #
# #
# class ViTEncoder(nn.Module):
#     def __init__(self, in_channels=3, features=32):
#         super(ViTEncoder, self).__init__()
#         self.features = features
#         self.vit = ViT(
#             image_size=256,
#             patch_size=16,
#             num_classes=256,
#             dim=features,
#             depth=12,
#             heads=12,
#             mlp_dim=features*4,
#             pool="cls",
#             channels=in_channels,
#         )
#         #self.linear = nn.Linear(features * 256 * 256, features * 256 * 256)

#     def forward(self, x):
#         x = self.vit(x)
#         #print("SHAPE after VIT forward:", x.shape)
#         #x = self.linear(x)
#         #print("Shape after VIT linear layer:", x.shape)
#         #x = torch.reshape(x, (16, self.features, 256, 256))
#         x = torch.reshape(x, (-1, self.features, 256, 256))
#         #print("Shape after VIT reshape:", x.shape)
#         return x

# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, init_features=32):
#         super(UNet, self).__init__()

#         features = init_features
#         self.encoder1 = ViTEncoder(in_channels, features)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = UNet._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Rest of the UNet architecture remains the same
#         self.bottleneck = UNet._block(
#             features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose2d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = UNet._block(
#             (features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose2d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = UNet._block(
#             (features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose2d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = UNet._block(
#             (features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose2d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = UNet._block(features * 2, features, name="dec1")

#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )

#     def forward(self, x):
#        # print("Starting shape:", x.shape)
#         enc1 = self.encoder1(x)
#        # print("SHAPE AFTER ENC1:", enc1.shape)
#         enc2 = self.encoder2(self.pool1(enc1))
#        # print("SHAPE AFTER ENC2:", enc2.shape)
#         enc3 = self.encoder3(self.pool2(enc2))
#        # print("SHAPE AFTER ENC3:", enc3.shape)
#         enc4 = self.encoder4(self.pool3(enc3))
#        # print("SHAPE AFTER ENC4:", enc4.shape)

#         bottleneck = self.bottleneck(self.pool4(enc4))

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         return torch.sigmoid(self.conv(dec1))

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                 ]
#             )
#         )


# from collections import OrderedDict
# import torch
# import torch.nn as nn

# class ReversedUNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, init_features=32):
#         super(ReversedUNet, self).__init__()

#         features = init_features

#         self.upconv1 = nn.ConvTranspose2d(
#             in_channels, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder1 = ReversedUNet._block(in_channels, features, name="dec1")
#         self.upconv2 = nn.ConvTranspose2d(
#             features, features, kernel_size=2, stride=2
#         )
#         self.decoder2 = ReversedUNet._block(features * 2, features, name="dec2")
#         self.upconv3 = nn.ConvTranspose2d(
#             features, features // 2, kernel_size=2, stride=2
#         )
#         self.decoder3 = ReversedUNet._block(features, features // 2, name="dec3")
#         self.upconv4 = nn.ConvTranspose2d(
#             features // 2, features // 4, kernel_size=2, stride=2
#         )
#         self.decoder4 = ReversedUNet._block(features // 2, features // 4, name="dec4")
#         self.conv = nn.Conv2d(
#             in_channels=features // 4, out_channels=out_channels, kernel_size=1
#         )

#     def forward(self, x):
#         upconv1 = self.upconv1(x)
#         dec1 = self.decoder1(upconv1)
#         upconv2 = self.upconv2(dec1)
#         dec2 = self.decoder2(torch.cat((upconv2, upconv1), dim=1))
#         upconv3 = self.upconv3(dec2)
#         dec3 = self.decoder3(torch.cat((upconv3, dec2), dim=1))
#         upconv4 = self.upconv4(dec3)
#         dec4 = self.decoder4(torch.cat((upconv4, dec3), dim=1))
#         return torch.sigmoid(self.conv(dec4))

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                 ]
#             )
#         )


from collections import OrderedDict
import torch
import torch.nn as nn
torch.cuda.empty_cache()

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=256):
        super(UNet, self).__init__()

        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features // 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features // 2, features // 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features // 4, features // 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(
            features // 8, features // 4, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features // 4, features // 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block(
            features // 8 * 2, features // 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features // 8, features // 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block(
            features // 4 * 2, features // 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features // 4, features // 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block(
            features // 2 * 2, features // 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features // 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features // 8, out_channels=out_channels, kernel_size=1
        )
        self.encoder5 = UNet._block(features, features // 8, name="enc4")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
       # print("shape before enc1: ", x.shape)
        enc1 = self.encoder1(x)
      #  print("shape before pool1: ", enc1.shape)
        pool1 = self.pool1(enc1)
       # print("shape before enc2: ", pool1.shape)
        enc2 = self.encoder2(pool1)

      #  print("shape before pool2: ", enc2.shape)
        pool2 = self.pool2(enc2)
      #  print("shape before enc3: ", pool2.shape)
        enc3 = self.encoder3(pool2)
      #  print("shape before pool3: ", enc3.shape)
        pool3 = self.pool3(enc3)
      #  print("shape before enc4: ", pool3.shape)
        enc4 = self.encoder4(pool3)
      #  print("shape before pool4: ", enc4.shape)
        pool4 = self.pool4(enc4)
      #  print("shape before bottleneck: ", pool4.shape)

        bottleneck = self.bottleneck(pool4)
       # print("shape before upconv4: ", bottleneck.shape)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
       # print("shape before upconv3: ", dec4.shape)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
       # print("shape before upconv2: ", dec3.shape)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
       # print("shape before dec2: ", dec2.shape)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        print("shape before last encoder: ", dec1.shape)
        dec0 = self.encoder5(dec1)
        # dec0 = self.pool5(dec0)
        print("shape before final conv and sigmoid: ",
              dec0.shape)  # 16, 256, 256, 256

        return torch.sigmoid(self.conv(dec0))

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


# Traceback (most recent call last):
#   File "train.py", line 255, in <module>
#     main(args)
#   File "train.py", line 62, in main
#     y_pred = unet(x)
#   File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 1102, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "/workspace/unet.py", line 356, in forward
#     dec1 = self.decoder1(dec1)
#   File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 1102, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/container.py", line 141, in forward
#     input = module(input)
#   File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 1102, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/conv.py", line 446, in forward
#     return self._conv_forward(input, self.weight, self.bias)
#   File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/conv.py", line 443, in _conv_forward
#     self.padding, self.dilation, self.groups)
# RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[16, 32, 512, 512] to have 3 channels, but got 32 channels instead
