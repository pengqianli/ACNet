import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

#########################################################
# ACNET VGG-16.
#########################################################


class Model(nn.Module):

    def __init__(self, model_path=None):
        super(Model, self).__init__()
    
        self.encoder = self.load_model(model_path) 
        
        self.condense_block1 = nn.Sequential(
            ConcentratedModule(1024, 256, 512, 32),
            AttentionPart(512)
        )
        
        self.condense_block2 = nn.Sequential(
            ConcentratedModule(512, 128, 256, 16),
            AttentionPart(256)
        )
        self.condense_block3 = nn.Sequential(
            ConcentratedModule(256, 64, 128, 8),
            AttentionPart(128)
        )
        
        self.outconv = nn.Conv2d(128, 1, 1)
        self.relu = nn.ReLU()
        self.param_init()
        print('create net successfully!')

    def _feature_extract(self, h, l):
        feature_map_h = self.encoder(h)
        feature_map_l = self.encoder(l)
        feature_map_l = F.interpolate(feature_map_l, size=(30, 40), mode='bilinear', align_corners=False)
        feature_map_l = torch.cat((feature_map_h, feature_map_l), 1)
        return feature_map_l
        
    def forward(self, h, l):
        feature_map = self._feature_extract(h, l)
        feature_map1, ca = self.condense_block1(feature_map)
        feature_map2, _ = self.condense_block2(feature_map1)
        feature_map3, _ = self.condense_block3(feature_map2)
        output = self.outconv(feature_map3)
        return output

    def load_model(sef, model_path=None):
        model = models.vgg16(pretrained=False)
        model.features[24] = nn.Conv2d(512, 512, 3, 1, 2, dilation=2)
        model.features[26] = nn.Conv2d(512, 512, 3, 1, 2, dilation=2)
        model.features[28] = nn.Conv2d(512, 512, 3, 1, 2, dilation=2)
        if (model_path != None):
            model.load_state_dict(torch.load(model_path))
        features = list(model.features)[:30]
        return nn.Sequential(*features)

    def param_init(self):

        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            nn.init.constant_(m.bias.data, 0.1)

        def xavier_init(m):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)

        xavier_init(self.outconv)


#############################New Attention####################################
class AttentionPart(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AttentionPart, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
        )
        self.max_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
        )
        self.final_fc = nn.Linear(channel // reduction, channel)
        # spatial attention
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, 1, 1),
            nn.BatchNorm2d(channel // 8),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(channel // 8),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, 1, 3, dilation=3),
            nn.BatchNorm2d(channel // 8),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(channel // 8 * 3, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.param_init()

    def forward(self, x):
        b, c, _, _ = x.size()
        # channel weight
        y = self.max_pool(x).view(b, c)
        y = self.max_fc(y)
        z = self.avg_pool(x).view(b, c)
        z = self.avg_fc(z)
        y = self.sigmoid(self.final_fc(y + z).view(b, c, 1, 1))
        # spatial weight
        z = torch.cat((self.conv3(x), self.conv5(x), self.conv7(x)), 1)
        z = self.sigmoid(self.final_conv(z))
        return x * y * z, y

    def param_init(self):

        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        def xavier_init(m):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)

        xavier_init(self.max_fc[0])
        xavier_init(self.avg_fc[0])
        xavier_init(self.final_fc)
        xavier_init(self.conv3[0])
        xavier_init(self.conv5[0])
        xavier_init(self.conv7[0])
        xavier_init(self.final_conv)


class ConcentratedModule(nn.Module):
    def __init__(self, in_channels, reduction, out_channels, group=32):
        super(ConcentratedModule, self).__init__()

        self.branch1_1x1_red = nn.Conv2d(in_channels, reduction, 1)
        self.branch2_1x1_red = nn.Conv2d(in_channels, reduction, 1)
        self.branch3_1x1_red = nn.Conv2d(in_channels, reduction, 1)
        self.branch1_3x3 = nn.Conv2d(reduction, reduction, 3, 1, 1, dilation=1, groups=group)
        self.branch1_5x5 = nn.Conv2d(reduction, reduction, 3, 1, 2, dilation=2, groups=group)
        self.branch1_7x7 = nn.Conv2d(reduction, reduction, 3, 1, 3, dilation=3, groups=group)
        self.branch1_1x1_fuse = nn.Conv2d(reduction, reduction, 1)
        self.branch2_1x1_fuse = nn.Conv2d(reduction, reduction, 1)
        self.branch3_1x1_fuse = nn.Conv2d(reduction, reduction, 1)
        self.branch1_1x1_out = nn.Conv2d(reduction, out_channels, 1)
        self.branch2_1x1_out = nn.Conv2d(reduction, out_channels, 1)
        self.branch3_1x1_out = nn.Conv2d(reduction, out_channels, 1)


        self.relu = nn.ReLU(inplace=True)
        self.param_init()
    
    def forward(self, x):
        # first branch
        f1 = self.relu(self.branch1_1x1_red(x))
        f1_1 = self.relu(self.branch1_3x3(f1) + f1)
        f1_2 = self.relu(self.branch1_1x1_fuse(f1_1))
        # second branch
        f2 = self.relu(self.branch2_1x1_red(x)) + f1_2
        f2_1 = self.relu(self.branch1_5x5(f2) + f2)
        f2_2 = self.relu(self.branch2_1x1_fuse(f2_1))
        # third branch
        f3 = self.relu(self.branch3_1x1_red(x)) + f2_2
        f3_1 = self.relu(self.branch1_7x7(f3) + f3)
        f3_2 = self.relu(self.branch3_1x1_fuse(f3_1))
        y = self.relu(self.branch1_1x1_out(f1_2) + self.branch2_1x1_out(f2_2) + self.branch3_1x1_out(f3_2))
        return y

    def param_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
