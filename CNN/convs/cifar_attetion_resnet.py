'''
Reference:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


        # layers = [nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
        #           nn.BatchNorm2d(planes),
        #           nn.ReLU(inplace=True),
        #           # nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
        #           # nn.BatchNorm2d(planes),
        #           # nn.ReLU(inplace=True),
        #           ]
        # self.side_layers = nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.BatchNorm1d(channel),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return y

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,  # stride=2 -> stride=1 for cifar
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Removed in _forward_impl for cifar
        dim = 64
        self.layer1 = self._make_layer(block, dim, layers[0])
        self.layer2 = self._make_layer(block, dim*2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, dim*4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, dim*8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = dim*8 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # Removed in _forward_impl

        self.fc =nn.Sequential(nn.Linear(dim*8 * block.expansion, dim*8 * block.expansion),
                               nn.BatchNorm1d(dim*8 * block.expansion),
                               nn.ReLU(inplace=True),
                               )

        # self.attention_net = attention_net()
        # self.append_attention_net()


        # self.f_conv = self._make_conv2d_layer(3, nf, max_pool=False, stride=1, padding=1)

        dim=64
        self.f_conv1 = self._make_conv2d_layer(3, dim, dim, max_pool=False,stride=1, padding=1)
        self.f_conv2 = self._make_conv2d_layer(dim * 1, (dim * 1), dim * 2, padding=1, stride=1, max_pool=True)
        self.f_conv3 = self._make_conv2d_layer(dim * 2, (dim * 2), dim * 4, padding=1, stride=1, max_pool=True)
        self.f_conv4 = self._make_conv2d_layer(dim * 4, (dim * 4), dim * 8, padding=1, stride=1, max_pool=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight!=None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    @staticmethod
    def _make_conv2d_layer(in_maps, mid_maps, out_maps, max_pool=False, stride=1, padding=1,frist_layer=False):

        layers = [nn.Conv2d(in_maps, mid_maps, kernel_size=3, stride=stride, padding=padding),
                  nn.BatchNorm2d(mid_maps),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(mid_maps, out_maps, kernel_size=3, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_maps),
                  nn.ReLU(inplace=True)
                  ]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        m0 = self.conv1(x)
        m0 = self.relu(self.bn1(m0))
        # h0 = self.f_conv1(x)
        h1 = self.f_conv1(x)
        h2 = self.f_conv2(h1)
        h3 = self.f_conv3(h2)
        h4 = self.f_conv4(h3)

        # h1_ = self.f_conv1_(x)
        # h2_ = self.f_conv2_(h1_)
        # h3_ = self.f_conv3_(h2_)
        # h4_ = self.f_conv4_(h3_)


        m1_= self.layer1(m0)#*self.f_conv1(m0)
        m1 = m1_ * h1

        m2_ = self.layer2(m1)#* self.f_conv2(m1)
        m2 = m2_ * h2

        m3_ = self.layer3(m2)#* self.f_conv3(m2)
        m3 = m3_ * h3

        m4_ = self.layer4(m3)#* self.f_conv4(m3)
        m4 = m4_ * h4


        # m4 = self.cnn(m4)

        pooled = self.avgpool(m4)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]

        # features = self.fc(features)
        return {
            'fmaps': [m1, m2, m3, m4],
            # 'mmaps':[m1_, m2_, m3_, m4_],
            # 'mask': [h1, h2, h3, h4],
            # 'neg_mask': [neg_h1_, neg_h2_, neg_h3_, neg_h4_],
            'features': features,
        }

    def _forward_impl_wo_mask(self, x):
        # # See note [TorchScript super()]

        m0 = self.conv1(x)
        m0 = self.relu(self.bn1(m0))

        m1 = self.layer1(m0)
        m2 = self.layer2(m1)
        m3 = self.layer3(m2)
        m4 = self.layer4(m3)

        pooled = self.avgpool(m4)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]

        return {
            'fmaps': [m1, m2, m3, m4],
            # 'mask': [h1, h2, h3, h4],
            # 'neg_mask': [neg_h1_, neg_h2_, neg_h3_, neg_h4_],
            'features': features
        }

    def forward(self, x, flag=False):
        if flag:
            return self._forward_impl_wo_mask(x)
        return self._forward_impl(x)

    def append_attention_net(self):
        self.attention_net.append_network()

    @property
    def last_conv(self):
        if hasattr(self.layer4[-1], 'conv3'):
            return self.layer4[-1].conv3
        else:
            return self.layer4[-1].conv2

class Mask_Net(nn.Module):
    def __init__(self, in_maps, out_maps, max_pool=False, stride=1, padding=1, start = False):
        super(Mask_Net, self).__init__()

        if start :
            layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=stride, padding=padding),
                      nn.BatchNorm2d(out_maps),
                      ]
        else:
            layers = [
                        nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=stride, padding=padding),
                      nn.BatchNorm2d(out_maps),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(out_maps, out_maps, kernel_size=3, stride=stride, padding=padding),
                      nn.BatchNorm2d(out_maps),
                      ]

        self.layers = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.ave_pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = max_pool

    def forward(self, x):
        mask_bf = self.layers(x)
        mask = self.relu(mask_bf)
        if self.max_pool:
            mask = self.max_pool_layer(mask)
        return mask

class attention_net(nn.Module):
    def __init__(self):
        super(attention_net, self).__init__()


        self.f_conv1_list = nn.ModuleList()
        self.f_conv2_list = nn.ModuleList()
        self.f_conv3_list = nn.ModuleList()
        self.f_conv4_list = nn.ModuleList()

        # self.f_conv1 = self._make_conv2d_layer(3, nf, max_pool=False,stride=1, padding=1, shape = 32)
        # self.f_conv2 = self._make_conv2d_layer(nf * 1, nf * 2, padding=1, max_pool=True, shape = 32)
        # self.f_conv3 = self._make_conv2d_layer(nf * 2, nf * 4, padding=1, max_pool=True, shape = 16)
        # self.f_conv4 = self._make_conv2d_layer(nf * 4, nf * 8, padding=1, max_pool=True, shape = 8)

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, stride=1, padding=1, shape=4):
        layers = [nn.Conv2d(in_maps, in_maps, kernel_size=3, stride=1, padding=padding),
                  nn.BatchNorm2d(in_maps),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_maps),
                  nn.ReLU(inplace=True)
                  ]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def append_network(self):
        nf = 64
        self.f_conv1_list.eval()
        self.f_conv2_list.eval()
        self.f_conv3_list.eval()
        self.f_conv4_list.eval()
        self.f_conv1_list.append(self._make_conv2d_layer(3, nf, max_pool=False,stride=1, padding=1, shape = 32))
        self.f_conv2_list.append(self._make_conv2d_layer(nf * 1, nf * 2, padding=1, max_pool=True, shape = 32))
        self.f_conv3_list.append(self._make_conv2d_layer(nf * 2, nf * 4, padding=1, max_pool=True, shape = 16))
        self.f_conv4_list.append(self._make_conv2d_layer(nf * 4, nf * 8, padding=1, max_pool=True, shape = 8))

        # if len(self.f_conv4_list)>1:
        #     self.f_conv1_list[-1].load_state_dict(self.f_conv1_list[-2].state_dict())
        #     self.f_conv2_list[-1].load_state_dict(self.f_conv2_list[-2].state_dict())
        #     self.f_conv3_list[-1].load_state_dict(self.f_conv3_list[-2].state_dict())
        #     self.f_conv4_list[-1].load_state_dict(self.f_conv4_list[-2].state_dict())

    def forward(self, x, task):
        h1 = self.f_conv1_list[task](x)
        h2 = self.f_conv2_list[task](h1)
        h3 = self.f_conv3_list[task](h2)
        h4 = self.f_conv4_list[task](h3)
        return h1, h2, h3, h4

    def forward_wo_grad(self, x, task):
        with torch.no_grad():
            h1 = self.f_conv1_list[task](x)
            h2 = self.f_conv2_list[task](h1)
            h3 = self.f_conv3_list[task](h2)
            h4 = self.f_conv4_list[task](h3)
        return h1, h2, h3, h4

    def get_add_mask(self, x):
        h1, h2, h3, h4 = 0,0,0,0
        count = len(self.f_conv1_list)
        for task in range(len(self.f_conv1_list)):
            # if task != count-1 :
            #     h1_, h2_, h3_, h4_= self.forward_wo_grad(x, task)
            # else:
            h1_, h2_, h3_, h4_ = self.forward(x, task)
            h1+=h1_
            h2+=h2_
            h3+=h3_
            h4+=h4_
        return h1/count, h2/count, h3/count, h4/count
        # return h1, h2, h3, h4

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
