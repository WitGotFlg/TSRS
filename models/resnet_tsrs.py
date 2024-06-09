import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import copy
import torch

__all__ = ['ResNet', 'resnet50_tsrs']


class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, opt, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features
        self.Ilayer = 2
        self.opt = opt

    def forward(self, x, test=True):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if test == True:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x
        else:
            noise_x = x
            noise_outputs_ = []
            for i, layer_module in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                if i == self.Ilayer:
                    noise = self.opt.noise * torch.randn(x.shape)
                    noise_x = x + noise.cuda(device=0)
                layer_outputs = layer_module(x)
                layer_noise_outputs = layer_module(noise_x)
                x = layer_outputs
                noise_x = layer_noise_outputs
                if i >= self.opt.Nlayer:
                    noise_stabling = torch.norm(noise_x - x, p=2) / torch.norm(x, p=2)
                    noise_outputs_.append(noise_stabling)
            x = layer_noise_outputs
            return x, noise_outputs_


    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def _resnet(arch, block, layers, pretrained, progress, opt, **kwargs):
    model = ResNet(opt, block, layers, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model



def resnet50_tsrs(pretrained=False, progress=True, opt=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, opt,
                   **kwargs)
