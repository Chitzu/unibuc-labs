import torch
from torch import nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.n_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, self.n_channels, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.n_channels),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_channels, 2 * self.n_channels, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2 * self.n_channels),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * self.n_channels, 4 * self.n_channels, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4 * self.n_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * self.n_channels, 8 * self.n_channels, 3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8 * self.n_channels),
            nn.Dropout(0.3)
        )

        ### Decoder
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(8 * self.n_channels, 4*self.n_channels, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*self.n_channels),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(4*self.n_channels, 2*self.n_channels, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*self.n_channels),
            nn.Dropout(0.3)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(2*self.n_channels, self.n_channels, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.n_channels)
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.n_channels, 2, 3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2),
            nn.Dropout(0.3)
        )
        
        self.convout = nn.Sequential(
            nn.Conv2d(2, 2, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.deconv1(x4) + x3
        x6 = self.deconv2(x5) + x2
        x7 = self.deconv3(x6) + x1
        x8 = self.deconv4(x7)
        mask = self.convout(x8)
        y = mask * x
        return y

