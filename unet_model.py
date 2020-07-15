import torch
import torch.nn as nn
from unet_parts import Down, Up, FC

from torch import Tensor

class ResUNet_3(nn.Module):
    """Residual U-Net which takes a volume of 3 slices and gives the 
    segmentation map for the middle slice. 
    
    This model is adopted from the paper: Deep learning to achieve clinically 
    applicable segmentation of head and neck anatomy for radiotherapy,
    https://arxiv.org/abs/1809.04430. 
    """
    def __init__(self) -> None:
        super(ResUNet_3, self).__init__()

        self.d1 = Down((3, 0, True), 1, 32, BN = True)
        self.d2 = Down((3, 0, True), 32, 32, BN = True)
        self.d3 = Down((3, 0, True), 32, 64, BN = True)
        self.d4 = Down((1, 0, True), 64, 64, BN = True)
        self.d5 = Down((1, 0, True), 64, 128, BN = True)
        self.d6 = Down((1, 0, True), 128, 128, BN = True)
        self.d7 = Down((1, 4, False), 128, 256, 3, 1)

        self.fc = FC(256, 1024, 256, 8)

        self.u7 = Up((4, False), 256*2, 256)
        self.u6 = Up((4, True), 3*128+256, 128, BN = True)
        self.u5 = Up((4, True), 3*128+128, 128, BN = True)
        self.u4 = Up((3, True), 3*64+128, 64, BN = True)
        self.u3 = Up((3, True), 3*64+64, 64, BN = True)
        self.u2 = Up((3, True), 3*32+64, 64, BN = True)
        self.u1 = Up((3, True), 3*32+64, 64, BN = True)

        self.segment = nn.Conv2d(64, 1, 1)

        self.last_activation = nn.Sigmoid()


    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.size()[0]

        x = x.view(-1, 1, 512, 512)

        conv1, out = self.d1(x, batch_size)
        conv2, out = self.d2(out, batch_size)
        conv3, out = self.d3(out, batch_size)
        conv4, out = self.d4(out, batch_size)
        conv5, out = self.d5(out, batch_size)
        conv6, out = self.d6(out, batch_size)
        conv7, out = self.d7(out, batch_size)

        out = self.fc(out)

        out = self.u7(out, conv7.view(batch_size, 256, 8, 8))
        out = self.u6(out, conv6.view(batch_size, 3*128, 16, 16))
        out = self.u5(out, conv5.view(batch_size, 3*128, 32, 32))
        out = self.u4(out, conv4.view(batch_size, 3*64, 64, 64))
        out = self.u3(out, conv3.view(batch_size, 3*64, 128, 128))
        out = self.u2(out, conv2.view(batch_size, 3*32, 256, 256))
        out = self.u1(out, conv1.view(batch_size, 3*32, 512, 512))

        out = self.last_activation(self.segment(out))

        return out
