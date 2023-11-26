from torch import nn, cat
from torch.nn.functional import relu

class UNet(nn.Module):
    def __init__(self, features=6, ):
        super().__init__()
        
        # Encoder
        self.enc11 = nn.Conv2d(features, 16, kernel_size=3, padding=1, padding_mode='zeros')
        self.enc12 = nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc21 = nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode='zeros')
        self.enc22 = nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='zeros')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc31 = nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode='zeros')
        self.enc32 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='zeros')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc41 = nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode='zeros')
        self.enc42 = nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode='zeros')

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec11 = nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='zeros')
        self.dec12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='zeros')

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec21 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='zeros')
        self.dec22 = nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode='zeros')

        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec31 = nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='zeros')
        self.dec32 = nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='zeros')
        
        # Output
        self.outconv = nn.Conv2d(16, features, kernel_size=1)

    def forward(self, X):
        # Encode
        xe11 = relu(self.enc11(X))
        xe12 = relu(self.enc12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.enc21(xp1))
        xe22 = relu(self.enc22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.enc31(xp2))
        xe32 = relu(self.enc32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.enc41(xp3))
        xe42 = relu(self.enc42(xe41))

        # Decode
        xuc1 = self.upconv1(xe42)
        xc1 = cat([xuc1, xe32], dim=1)
        xd11 = relu(self.dec11(xc1))
        xd12 = relu(self.dec12(xd11))

        xuc2 = self.upconv2(xd12)
        xc2 = cat([xuc2, xe22], dim=1)
        xd21 = relu(self.dec21(xc2))
        xd22 = relu(self.dec22(xd21))

        xuc3 = self.upconv3(xd22)
        xc3 = cat([xuc3, xe12], dim=1)
        xd31 = relu(self.dec31(xc3))
        xd32 = relu(self.dec32(xd31))

        # Out
        out = self.outconv(xd32)
        return out
