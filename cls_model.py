import torch
import torch.nn as nn


class TemplateNetwork(nn.Module):
    def __init__(self):
        super(TemplateNetwork, self).__init__()
        self.features = \
            nn.Sequential(\
                # 1
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(15, 4)),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=(10, 1)),
                # 2
                nn.Conv2d(in_channels=10, out_channels=2, kernel_size=(2, 1)),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                )

        self.classifier = \
            nn.Sequential(
                nn.Linear(in_features=16, out_features=1),
                nn.Sigmoid(),
                nn.Dropout(p=0.9),
                nn.Linear(in_features=1, out_features=2))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn((1, 1, 196, 4))
    model = TemplateNetwork()
    out = model(x)
    print(out.shape)
