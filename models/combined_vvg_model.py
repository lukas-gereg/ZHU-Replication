import torch
import numpy as np
import torch.nn as nn
import torchvision


class CombinedVVGModel(nn.Module):
    def __init__(self, color_channels: int, image_size: tuple[int, int], pooling_method_constructor):
        super(self.__class__, self).__init__()

        self.vvg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.vvg16.avgpool = pooling_method_constructor((2, 2))
        self.vvg16.classifier = nn.Identity()

        self.vvg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.vvg19.avgpool = pooling_method_constructor((2, 2))
        self.vvg19.classifier = nn.Identity()

        self.dropout = nn.Dropout(0.5)

        fc_input = self._calculate_fc_input_size(color_channels, image_size)

        self.fc1 = nn.Sequential(nn.Linear(in_features=fc_input, out_features=1000), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(in_features=1000, out_features=500), nn.Sigmoid())
        self.fc3 = nn.Sequential(nn.Linear(in_features=500, out_features=150), nn.Sigmoid())
        self.fc4 = nn.Sequential(nn.Linear(in_features=150, out_features=2), nn.Softmax(dim=1))

    def forward(self, x):
        x1 = self.vvg16(x)
        x2 = self.vvg19(x)

        x = torch.cat((x1, x2), dim=1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc4(self.fc3(self.fc2(x)))

        return x

    def _calculate_fc_input_size(self, color_channels: int, image_size: tuple[int, int]):
        batch_vvg16, *dims_vvg16 = self.vvg16(torch.rand(1, color_channels, *image_size)).shape
        batch_vvg19, *dims_vvg19 = self.vvg19(torch.rand(1, color_channels, *image_size)).shape

        return np.prod(dims_vvg16) + np.prod(dims_vvg19)