import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, model):
        super(Autoencoder, self).__init__()

        # Encoder: Use ResNet-34 pre-trained model
        self.encoder = model(weights="DEFAULT")
        # Remove the fully connected layers (classification head)
        modules = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*modules)

        # Downsampling Sequential
        self.downsample = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):      # input  of [batch_size, 3, 384, 992]
        x = self.encoder(x)    # output of [batch_size, 512, 12, 31]
        x = self.downsample(x) # output of [batch_size, 32, 12, 31]
        x = self.decoder(x)    # output of [batch_size, 3, 384, 992]
        return x
