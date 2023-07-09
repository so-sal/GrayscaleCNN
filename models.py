import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class EfficientNetB3(nn.Module):
    def __init__(self, n_classes, input_channel=1, p=0.2):
        super(EfficientNetB3, self).__init__()
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.layers = [i for i in self.model.children()]
        
        if input_channel != 3:
            self.conv1 = nn.Conv2d(input_channel, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            first_conv_block = [i for i in self.layers[0][0]]
            self.first_conv_block = nn.Sequential(
                self.conv1,
                *first_conv_block[1:])
            
            self.model = nn.Sequential(
                self.first_conv_block,
                *[i for i in self.layers[0].children()][1:],
                *self.layers[1:-1]
            )
        else:
            self.model = nn.Sequential(
                *[i for i in self.model.children()][:-1]
            )
        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(1536, n_classes)
        )
        self.gradients = None
              
    def forward(self, x):
        out = self.model(x)
        
        if out.requires_grad:
            h = out.register_hook(self.activations_hook)        

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out     

    
class EfficientNetB0(nn.Module):
    def __init__(self, n_classes, input_channel=1, p=0.2):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.layers = [i for i in self.model.children()]
        
        if input_channel != 3:
            self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            first_conv_block = [i for i in self.layers[0][0]]
            self.first_conv_block = nn.Sequential(self.conv1,*first_conv_block[1:])
            self.model = nn.Sequential(
                self.first_conv_block,
                # *[i for i in self.layers[0][0].children()][1:],
                *[i for i in self.layers[0].children()][1:],
                *self.layers[1:-1]
            )
        else:
            self.model = nn.Sequential(
                *[i for i in self.model.children()][:-1]
            )
        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(1280, n_classes)
        )
        self.gradients = None

    def forward(self, x):
        out = self.model(x)

        if out.requires_grad:
            h = out.register_hook(self.activations_hook) 
        
        out = torch.flatten(out, 1)        
        out = self.classifier(out)
        return out        


class ResNet50Scratch(nn.Module):
    def __init__(self, n_classes, input_channel=1):
        super(ResNet50Scratch, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, num_classes=n_classes)
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.layers = [i for i in self.model.children()]
        self.model = nn.Sequential(self.conv1,
                                   *self.layers[1:-1])
        self.classifier = self.layers[-1]
        
    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
