# GrayscaleCNN
A pretrained CNN model that takes input in grayscale.<br>
In a typical CNN model that usually takes input in RGB format with dimensions (224, 224, 3), there are image when you may want to provide input in grayscale format with dimensions (224, 224, 1).<br>

One approach to accommodate grayscale input is to modify the first Convolutional layer to accept only a single channel as the input, instead of three.<br>
<br>
<br>
<br>
## Library import
from torchvision.models import resnet50, ResNet50_Weights<br>
import torch<br>
import torch.nn as nn<br>
import torch.nn.functional as F<br>
import torchvision.transforms as transforms<br>
import numpy as np<br>
<br>
<br>
## define models
GrayResNet50 = models.GrayResNet50(n_classes=3, input_channel=1)<br>
GrayEffNetB0 = models.GrayEfficientNetB0(n_classes=3, input_channel=1)<br>
GrayEffNetB3 = models.GrayEfficientNetB3(n_classes=3, input_channel=1)<br>
<br>
<br>
## example input data (224, 224, 1)
trans = transforms.Compose([<br>
    transforms.ToPILImage(),<br>
    transforms.GaussianBlur(kernel_size=3),<br>
    transforms.ToTensor()<br>
])<br>
tmp_image = np.random.randint(0, 256, size=(224, 224, 1), dtype=np.uint8)<br>
input_tensor = trans(tmp_image).float().unsqueeze(0)<br>
<br>
<br>
## inference
out1 = GrayResNet50(input_tensor)<br>
out2 = GrayEffNetB0(input_tensor)<br>
out3 = GrayEffNetB3(input_tensor)<br>
