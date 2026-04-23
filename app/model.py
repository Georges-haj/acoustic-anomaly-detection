import torch.nn as nn
import torch.nn.functional as F
from app.config import N_MACHINE_TYPES, CONFIG


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)


class ConvTBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class MultiTaskAnomalyModel(nn.Module):
    def __init__(self, num_machine_types=N_MACHINE_TYPES,
                 latent_dim=CONFIG["latent_dim"],
                 dropout=CONFIG["dropout"]):
        super().__init__()
        self.num_machine_types = num_machine_types

        self.encoder = nn.Sequential(
            ConvBlock(  1,  32),
            ConvBlock( 32,  64),
            ConvBlock( 64, 128),
            ConvBlock(128, 256),
        )
        self.decoder = nn.Sequential(
            ConvTBlock(256, 128),
            ConvTBlock(128,  64),
            ConvTBlock( 64,  32),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_machine_types),
        )

    def forward(self, x):
        z         = self.encoder(x)
        recon     = self.decoder(z)
        recon     = F.interpolate(recon, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)
        mt_logits = self.classifier(z)
        return recon, mt_logits, z

    def anomaly_score(self, x):
        recon, _, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])

    def predict_machine_type(self, x):
        _, logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_latent(self, x):
        return self.encoder(x).mean(dim=[2, 3])