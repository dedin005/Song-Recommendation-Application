import torch
import torch.nn as nn
from torchvision import models


class VGG16Autoencoder(nn.Module):
    def __init__(self, _):
        super(VGG16Autoencoder, self).__init__()
        # Load pre-trained VGG16 model + higher level layers
        vgg16 = models.vgg16(weights="DEFAULT")

        # Encoder: Use the features from VGG16
        self.encoder_features = vgg16.features
        self.encoder_classifier = nn.Sequential(
            nn.Linear(512*31*12, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512*31*12),
            nn.ReLU(True)
        )
        self.decoder_features = nn.Sequential(
            # Add layers to upscale to the input image size (512 -> 3 channels)
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()  # Normalize the output to [0,1]
        )

    def forward(self, x):
        # Encoder
        x = self.encoder_features(x)
        x = torch.flatten(x, 1)
        x = self.encoder_classifier(x)

        # Decoder
        x = self.decoder_classifier(x)
        x = x.view(-1, 512, 12, 31)  # Reshape to match the feature map size
        x = self.decoder_features(x)
        return x
