import net.pspnet as pspnet
import net.deeplab as deeplab


def get_deeplab(encoder, encoder_dim, classes):
    return deeplab.DeepLab(encoder=encoder, encoder_dim=encoder_dim, classes=classes)


def get_pspnet(encoder, encoder_dim, classes):
    return pspnet.PSPNet(encoder=encoder, encoder_dim=encoder_dim, classes=classes)

