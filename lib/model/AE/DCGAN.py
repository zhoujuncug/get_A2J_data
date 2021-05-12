import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1 , 64 , 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256 , 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(2048),
            nn.ReLU(True),
            nn.Conv2d(2048, 256 , 3, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 2048, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(2048),
            nn.ReLU(True),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 1, 1, 0, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2048, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
    
        return output.view(-1, 1).squeeze(1)