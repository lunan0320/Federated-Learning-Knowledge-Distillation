from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models


class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
        super(EncoderFemnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320), code_length)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        z = F.relu(self.fc1(x))
        return z       
        
class CNNFemnist(nn.Module):
    def __init__(self, args,code_length=50,num_classes = 62):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return p
       
        
class ResNet18(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ResNet18, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes  
        self.feature_extractor = models.resnet18(num_classes=self.code_length)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.code_length, self.num_classes))
    def forward(self,x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return p
    
class ShuffLeNet(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ShuffLeNet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes  
        self.feature_extractor = models.shufflenet_v2_x1_0(num_classes=self.code_length)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.code_length, self.num_classes))
    def forward(self,x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return p