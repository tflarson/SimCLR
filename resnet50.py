import torchvision
import torch.nn as nn
def ResNet50():
    model = torchvision.models.resnet50()
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = conv1
    model.maxpool = nn.Identity()
    return model


