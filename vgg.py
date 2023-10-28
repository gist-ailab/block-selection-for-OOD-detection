'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        self.classifier = nn.Sequential(                             
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512 , 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        #  # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_() # for bias

    def forward(self, x):
        out = self.forward_features(x)
        out = self.classifier(out)
        return out

    def forward_features(self,x):
        out = self.features(x)
        return out.view(out.size(0), -1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1 ,bias=True), #bias=False
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
    def forward_features_blockwise(self, x):
        # VGG11 forward features
        features = []

        x = self.features[0](x)        
        x = self.features[1](x); features.append(x) 
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x); features.append(x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x); features.append(x)
        x = self.features[8](x)
        x = self.features[9](x); features.append(x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x); features.append(x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x); features.append(x)
        x = self.features[16](x)
        x = self.features[17](x); features.append(x)
        x = self.features[18](x)
        x = self.features[19](x); features.append(x)

        return features

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()