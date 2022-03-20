import torch
import torch.nn as nn
from utils import util


class Discriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, class_label):
        inputs = self.conv1(inputs)
        inputs = self.leaky_relu(inputs)
        inputs = self.conv2(inputs)
        inputs = self.leaky_relu(inputs)
        inputs = self.conv3(inputs)
        inputs = self.leaky_relu(inputs)
        inputs = self.conv4(inputs)
        inputs = self.leaky_relu(inputs)
        inputs = self.classifier(inputs)
        labels = util.get_target_tensor(inputs, class_label).cuda()
        return self.loss(inputs, labels)


def get_dcgan(num_classes, **kwargs):
    return Discriminator(num_classes=num_classes)
