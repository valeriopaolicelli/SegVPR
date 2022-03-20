import torch
from torch import nn
import net.functional as func


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x, x_size):
        out = [x]
        for f in self.features:
            out.append(func.interpol(f(x), x_size))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, encoder=None, encoder_dim=2048, classes=17):
        super(PSPNet, self).__init__()
        self.bins = (1, 2, 3, 6)
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        fea_dim = encoder_dim
        self.ppm = PPM(fea_dim, int(fea_dim / len(self.bins)), self.bins)
        fea_dim = encoder_dim*2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, classes, kernel_size=1)
        )

    def forward(self, x, no_classifier=False):
        x_size = x.size()
        multiscale_feat = self.encoder(x)
        if no_classifier:
            return multiscale_feat, None
        final_sem = self.ppm(multiscale_feat[-1], multiscale_feat[-1].size()[2:])
        final_sem = self.cls(final_sem)
        final_sem = func.interpol(x=final_sem, size=(x_size[2], x_size[3]))
        return multiscale_feat, final_sem
