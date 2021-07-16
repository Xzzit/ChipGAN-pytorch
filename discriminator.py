import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (4, 4), stride, (1, 1), bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
            )
    
        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride=1, padding=1, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    real = torch.randn((5, 3, 256, 256))
    fake = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    pred_real = model(real)
    pred_fake = model(fake)
    print(pred_real.mean().item(), pred_fake.mean().item())


if __name__ == "__main__":
    test()