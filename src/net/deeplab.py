from torch import nn
import net.functional as func


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class DeepLab(nn.Module):
    def __init__(self, encoder=None, encoder_dim=2048, classes=17):
        super(DeepLab, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.cls = ClassifierModule(self.encoder_dim, [6, 12, 18, 24], [6, 12, 18, 24], classes)

    def forward(self, x, no_classifier=False):
        x_size = x.size()
        multiscale_feat = self.encoder(x)
        if no_classifier:
            return multiscale_feat, None
        final_sem = self.cls(multiscale_feat[-1])
        final_sem = func.interpol(x=final_sem, size=(x_size[2], x_size[3]))
        return multiscale_feat, final_sem
