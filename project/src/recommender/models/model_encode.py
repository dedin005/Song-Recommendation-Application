import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()

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

    def forward(self, x):      # input  of [batch_size, 3, 384, 992]
        x = self.encoder(x)    # output of [batch_size, 512, 12, 31]
        x = self.downsample(x) # output of [batch_size, 32, 12, 31]
        return x

