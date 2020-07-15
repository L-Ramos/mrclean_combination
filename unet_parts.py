import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

def m_conv2D(in_channels: int, out_channels: int, m: int, last_relu: bool = True, BN: bool = False)-> nn.Sequential:
    """Creates a 2D residual convolution block, filter size of 3 and stride 1, 
    for the given parameters. 
    
    Consult paper for more details: https://arxiv.org/abs/1809.04430, page 17.
    
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        m (int): The number 2D convolution kernels to be included. 
        last_relu (bool, optional): Defaults to True. Determines whether to 
            include ReLU activation layer as the last layer of the block.
        BN (bool, optional): Defaults to False. Determines whether to 
            include batch normalisation in the block.
    
    Returns:
        nn.Sequential() object in accordance with the input parameters.
    """
    m_conv = nn.Sequential()

    idx = 0
    for i in range(m):
        if i == 0:
            m_conv.add_module(str(idx), nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
        else:
            m_conv.add_module(str(idx), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
        idx += 1
        if i < m-1:
            if BN:
                m_conv.add_module(str(idx), nn.BatchNorm2d(out_channels))
                idx += 1
            m_conv.add_module(str(idx), nn.ReLU(inplace=True))
        elif  i == m-1 and last_relu:
            if BN:
                m_conv.add_module(str(idx), nn.BatchNorm2d(out_channels))
                idx += 1
            m_conv.add_module(str(idx), nn.ReLU(inplace=True))
        idx += 1

    return m_conv

def n_conv3D(in_channels: int, out_channels: int, n: int)-> nn.Sequential:
    """Creates a 3D residual convolution block, filter size of 3 and stride 1, 
    for the given parameters. 
    
    Consult paper for more details: https://arxiv.org/abs/1809.04430, page 17.
    
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        n (int): The number 3D convolution kernels to be included. 
    
    Returns:
        nn.Sequential() object in accordance with the input parameters.
    """
    n_conv = nn.Sequential()

    idx = 0
    for i in range(n):
        if i == 0:
            n_conv.add_module(str(idx), nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False))
        else:
            n_conv.add_module(str(idx), nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False))
        idx += 1
        if i != n-1:
            n_conv.add_module(str(idx), nn.ReLU(inplace=True))
        idx += 1

    return n_conv


class Down(nn.Module):
    """Down module consists of 2D and/or 3D convolutional blocks. They represent 
    a single stage in encoder part of the U-Net.

    Consult paper for more details: https://arxiv.org/abs/1809.04430, page 17.
    """
    def __init__(self, parameters: Tuple[int, int, bool], inplanes: int, planes: int, 
        inplanes3D: Optional[int] =None, planes3D: Optional[int]=None, BN: bool= False) -> None:
        """Constructs the 2D and/or 3D convolutional blocks based on the input
        arguments.
        
        Args:
            parameters (tuple): (m, n, pool)
                m (int): The number 2D convolution kernels to be included.
                n (int): The number 3D convolution kernels to be included. 
                pool (bool): Determines whether pooling operation has to be 
                    applied in the end or not.
            inplanes (int): The number of input channels for 2D conv block.
            planes (int): The number of ouput channels for 2D conv block.
            inplanes3D (int, optional): Defaults to None. The number of input 
                channels for 3D conv block. When None, the module doesn't have 
                3D conv block.
            planes3D (int, optional): Defaults to None. The number of output 
                channels for 3D conv block. When None, the module doesn't have 
                3D conv block.
            BN (bool, optional): Defaults to False. When true, batch normalisation 
                is added to the 2D conv blocks. 
        """
        super(Down, self).__init__()

        self.m, self.n, self.pool = parameters

        last_relu = False
        if self.n:
            last_relu = True

        self.conv2D = m_conv2D(inplanes, planes, self.m, last_relu = last_relu, BN = BN)
        self.conv2D_iden = nn.Conv2d(inplanes, planes, 1, bias=False)

        if self.n:
            self.conv3D = n_conv3D(inplanes3D, planes3D, self.n)
            self.conv3D_iden = nn.Conv3d(inplanes3D, planes3D, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        #maybe add batchnorm
        if self.pool:
            self.downsample = nn.MaxPool2d(2)

        self.inplanes = inplanes
        self.planes = planes
        self.inplanes3D = inplanes3D
        self.planes3D = planes3D

    def forward(self, x: Tensor, batch_size: int) -> Tuple[Tensor, Tensor]:
        """        
        Args:
            x (torch.Tensor): input image
            batch_size (int): Batch size
        
        Returns:
            conv_out (torch.Tensor): The processed output to be contactented 
                with the input of the corresponding stage of the decoder.
            pool_out (torch.Tensor): Downsampled output to be used as input for 
                the next layer. If pool is False then conv_out and pool_out are 
                the same.  
        """
        identity = x


        out = self.conv2D(self.relu(x))
        identity_out = self.conv2D_iden(identity)

        channels_dims = out.size()
        if self.n:
            out = out.view(batch_size, -1, channels_dims[1], channels_dims[2], channels_dims[3])
            out = self.conv3D(out)
            out = out.view(-1, channels_dims[1], channels_dims[2], channels_dims[3])

            id_ch_dims = identity_out.size()
            identity_out = identity_out.view(batch_size, -1, id_ch_dims[1], id_ch_dims[2], id_ch_dims[3])
            identity_out = self.conv3D_iden(identity_out)
            identity_out = identity_out.view(-1, id_ch_dims[1], id_ch_dims[2], id_ch_dims[3])

        out += identity_out

        conv_out = out.view(batch_size, -1, channels_dims[1], channels_dims[2], channels_dims[3])
        if self.pool:
            pool_out = self.downsample(out)
        else:
            pool_out = out

        return conv_out, pool_out

class FC(nn.Module):
    """FC module is the fully connected module between the encoder and the 
    decoder of the U-Net. 

    Consult paper for more details: https://arxiv.org/abs/1809.04430, page 17.
    """
    def __init__(self, inplanes: int, planes: int, outplanes:int , outkernel: int) -> None:
        """Constructs the fully connected module based on the input arguments.
        
        Args:
            inplanes (int): The number of input channels for the input 2D conv kernel.
            planes (int): The number of output channels for the input 2D conv kernel.
            outplanes (int): The number of channels in the input of the 
                next stage conv block.
            outkernel (int): The size of the feature maps which are input to the
                next stage conv block.
        """
        super(FC, self).__init__()

        self.outplanes = outplanes
        self.outkernel = outkernel

        self.conv = nn.Conv2d(inplanes, planes, outkernel, bias=False)
        self.fc1 = nn.Linear(planes, planes)
        self.fc2 = nn.Linear(planes, planes)
        self.fc3 = nn.Linear(planes, outkernel*outkernel*outplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Input image
        
        Returns:
            torch.Tensor: The output image with size (B, outplanes, outkernel, outkernel)
        """
        x = self.conv(x).view(-1, 1024)

        identity = x

        out = self.fc1(self.relu(x))
        out += identity

        identity = out

        out = self.fc2(self.relu(out))
        out += identity

        out = self.fc3(self.relu(out))

        return out.view(-1, self.outplanes, self.outkernel, self.outkernel)


class Up(nn.Module):
    """Up module consists of 2D convolutional blocks. They represent 
    a single stage in decoder part of the U-Net.

    Consult paper for more details: https://arxiv.org/abs/1809.04430, page 17.
    """
    def __init__(self, parameters: Tuple[int, bool], inplanes: int, planes: int, BN: bool = False) -> None:
        """Constructs the 2D convolutional blocks based on the input
        arguments.
        
        Args:
            parameters (tuple): (m, upscale)
                m (int): The number 2D convolution kernels to be included.
                upscale (bool): Determines whether upsampling operation has to be 
                    applied in the end or not.
            inplanes (int): The number of input channels for 2D conv block.
            planes (int): The number of output channels for 2D conv block.
            BN (bool, optional): Defaults to False. When true, batch normalisation 
                is added to the 2D conv blocks. 
        """
        super(Up, self).__init__()

        self.m, self.upscale = parameters

        self.conv = m_conv2D(inplanes, planes, self.m, last_relu=False, BN = BN)
        self.conv_iden = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        if self.upscale:
            self.upsample = nn.functional.interpolate

    def forward(self, x: Tensor, concat_planes: Optional[Tensor]=None) -> Tensor:
        """
        Args:
            x (torch.Tensor): Input image
            concat_planes (torch.Tensor, optional): Defaults to None. Filter maps
            from the similar stage of the encoder to be conctented with the input.
        
        Returns:
            torch.Tensor: The output image
        """
        if self.upscale:
            x = self.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)

        if type(concat_planes) == torch.Tensor:
            x = torch.cat([x, concat_planes], dim=1)

        identity = x

        out = self.conv(self.relu(x))
        identity_out = self.conv_iden(identity)

        out += identity_out

        return out
