from torch import nn, cat
from torch.nn.functional import relu
from models.config import BATCH_SIZE, INPUT_DIMS


class UNet(nn.Module):
    def __init__(
        self,
        features=6,
        spatial_features=6,
        temporal_features=4,
        out_features=6,
        s=3,
        fh=2,
        base_units=16,
    ):
        super().__init__()
        BASE = base_units
        self.lat, self.lon = INPUT_DIMS
        self.features = features
        self.mlp_embedder = nn.Linear(s * features, BASE)
        self.temporal_embedder = nn.Linear(temporal_features, self.lat * self.lon)
        enc11_input_size = BASE + 1 + spatial_features

        # Encoder
        self.enc11 = nn.Conv2d(
            enc11_input_size, BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.enc12 = nn.Conv2d(
            BASE, BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc21 = nn.Conv2d(
            BASE, 2 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.enc22 = nn.Conv2d(
            2 * BASE, 2 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc31 = nn.Conv2d(
            2 * BASE, 4 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.enc32 = nn.Conv2d(
            4 * BASE, 4 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc41 = nn.Conv2d(
            4 * BASE, 8 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.enc42 = nn.Conv2d(
            8 * BASE, 8 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc51 = nn.Conv2d(
            8 * BASE, 16 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.enc52 = nn.Conv2d(
            16 * BASE, 16 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )

        # Decoder
        self.upconv0 = nn.ConvTranspose2d(16 * BASE, 8 * BASE, kernel_size=2, stride=2)
        self.dec01 = nn.Conv2d(
            16 * BASE, 8 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.dec02 = nn.Conv2d(
            8 * BASE, 8 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )

        self.upconv1 = nn.ConvTranspose2d(8 * BASE, 4 * BASE, kernel_size=2, stride=2)
        self.dec11 = nn.Conv2d(
            8 * BASE, 4 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.dec12 = nn.Conv2d(
            4 * BASE, 4 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )

        self.upconv2 = nn.ConvTranspose2d(4 * BASE, 2 * BASE, kernel_size=2, stride=2)
        self.dec21 = nn.Conv2d(
            4 * BASE, 2 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.dec22 = nn.Conv2d(
            2 * BASE, 2 * BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )

        self.upconv3 = nn.ConvTranspose2d(2 * BASE, BASE, kernel_size=2, stride=2)
        self.dec31 = nn.Conv2d(
            2 * BASE, BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.dec32 = nn.Conv2d(
            BASE, BASE, kernel_size=3, padding=1, padding_mode="reflect"
        )

        # Output
        self.outconv = nn.Conv2d(BASE, fh * out_features, kernel_size=1)

    def forward(self, X, t, s, *args):
        # print(f"INPUT: X:{X.shape}, t:{t.shape}, s:{s.shape}")
        batch_size = X.shape[0]
        x = X.permute((0, 2, 3, 1)).reshape((batch_size, -1, X.shape[1]))
        # print(f"X new shape: {x.shape}")
        # Embed features
        Xe = relu(self.mlp_embedder(x))
        # print(f"X embedding: {Xe.shape}")
        te = (
            self.temporal_embedder(t.reshape(batch_size, 1, -1))
            .relu()
            .permute((0, 2, 1))
        )
        # print(f"t embeding: {te.shape}")
        concat = cat((Xe, te, s), dim=-1)
        # print(f"Concat before reshape: {concat.shape}")
        concat = concat.permute((0, 2, 1))
        concat = concat.reshape(
            concat.shape[:-1]
            + (
                self.lat,
                self.lon,
            )
        )
        # print(f"What goes into unet: {concat.shape}")
        # Encode
        xe11 = relu(self.enc11(concat))
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
        xp4 = self.pool4(xe42)

        xe51 = relu(self.enc51(xp4))
        xe52 = relu(self.enc52(xe51))

        # Decode
        xuc0 = self.upconv0(xe52)
        xc0 = cat([xuc0, xe42], dim=1)
        xd01 = relu(self.dec01(xc0))
        xd02 = relu(self.dec02(xd01))

        xuc1 = self.upconv1(xd02)
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
        # print(f"What goes out: {out.shape}")
        return out
