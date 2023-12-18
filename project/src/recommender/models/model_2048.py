import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, model):
        super(Autoencoder, self).__init__()

        # Encoder: Use ResNet-101 pre-trained model
        self.encoder = model(weights="DEFAULT")
        # Remove the fully connected layers (classification head)
        modules = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*modules)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)  # [batch, 3, 384, 992] = batch * 1142784
        x = self.encoder(x)
        # print(x.shape)  # [batch, 2048, 12, 31] = batch * 761856
        x = self.decoder(x)
        return x
