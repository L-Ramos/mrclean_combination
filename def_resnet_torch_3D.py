# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:55:19 2020

@author: laramos
"""


from IPython.display import SVG
#from tensorflow.keras.utils.vis_utils import model_to_dot
#from tensorflow.keras.utils import plot_model
#from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from tensorflow.keras import optimizers  
import random as python_random
import tensorflow.keras.backend as K
import random as python_random
import numpy as np
import tensorflow as tf

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)
import torch
import torch.nn as nn
import torch.nn.functional as F

#nn.BatchNorm2d
norm = nn.InstanceNorm3d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,use_clin, num_classes=1):
        super(ResNet, self).__init__()
        self.use_clin = use_clin
        self.in_planes = 8

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
       # self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.linear = nn.Linear(128*8*8, num_classes)
        
        if self.use_clin:
           self.linear = nn.Linear((64*8*8)+58, num_classes)
        else:
           self.linear = nn.Linear((64*8*8), num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,data=[]):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        if self.use_clin:
            data = data.view(data.size(0), -1)
            out = torch.cat((out, data), dim = 1)
        out = (self.linear(out))
        #out = (self.linear(out))
        out = self.sigmoid(out)
        return out


def ResNet18_3D(use_clin):
    return ResNet(BasicBlock, [2, 2, 2, 2],use_clin)


def ResNet34_3D(use_clin):
    return ResNet(BasicBlock, [3, 4, 6, 3],use_clin)



