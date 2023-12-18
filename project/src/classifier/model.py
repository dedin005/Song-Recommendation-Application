import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, model, model_output_size, num_classes=10):
        super(Classifier, self).__init__()

        self.resnet = model(weights="DEFAULT")  # Use pretrained models

        # Remove the last fully connected layer (for 1000 classes)
        # Note: The last layer is named 'fc' in ResNet models
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Add a new head to handle your specific input size and number of classes
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(model_output_size, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.head(x)
        return x
