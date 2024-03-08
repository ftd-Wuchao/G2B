import copy
from itertools import chain
import math

import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34
from convs.cifar_resnet18 import resnet18 as cifar_resnet18
from convs.cifar_resnet18 import resnet34 as cifar_resnet34
from convs.cifar_resnet18 import resnet50 as cifar_resnet50
from convs.cifar_wide_resnet import WideResNet_16_1, WideResNet_16_2, WideResNet_16_4, WideResNet_16_8, WideResNet_16_16
from convs.cifar_wide_resnet import WideResNet_Attention_16_1, WideResNet_Attention_16_2, WideResNet_Attention_16_4, WideResNet_Attention_16_8, WideResNet_Attention_16_16
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from convs.attention_resnet import resnet18 as attention_resnet18
from convs.cifar_attetion_resnet import resnet18 as cifar_attention_resnet18
from convs.cifar_attetion_resnet import resnet34 as cifar_attention_resnet34


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18()
    elif name == 'resnet34':
        return resnet34()
    elif name == 'attention_resnet18':
        return attention_resnet18()
    elif name == 'cifar_attention_resnet18':
        return cifar_attention_resnet18()
    elif name == 'cifar_attention_resnet34':
        return cifar_attention_resnet34()
    elif name == 'cifar_resnet18':
        return cifar_resnet18()
    elif name == 'cifar_resnet34':
        return cifar_resnet34()
    elif name == 'cifar_resnet50':
        return cifar_resnet50()
    elif name == 'wideresnet_16_1':
        return WideResNet_16_1()
    elif name == 'wideresnet_16_2':
        return WideResNet_16_2()
    elif name == 'wideresnet_16_4':
        return WideResNet_16_4()
    elif name == 'wideresnet_16_8':
        return WideResNet_16_8()
    elif name == 'wideresnet_16_16':
        return WideResNet_16_16()
    elif name == 'wideresnet_attention_16_1':
        return WideResNet_Attention_16_1()
    elif name == 'wideresnet_attention_16_2':
        return WideResNet_Attention_16_2()
    elif name == 'wideresnet_attention_16_4':
        return WideResNet_Attention_16_4()
    elif name == 'wideresnet_attention_16_8':
        return WideResNet_Attention_16_8()
    elif name == 'wideresnet_attention_16_16':
        return WideResNet_Attention_16_16()
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {'fmaps': [x_1, x_2, ..., x_n],
        'features': features
        'logits': logits}
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_paras(self, params):
        for param in params.parameters():
            param.requires_grad = False
        self.eval()

    def freeze_mask(self):
        self.freeze_paras(self.convnet.f_conv1)
        self.freeze_paras(self.convnet.f_conv2)
        self.freeze_paras(self.convnet.f_conv3)
        self.freeze_paras(self.convnet.f_conv4)


class IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        self.bn = nn.BatchNorm1d(self.feature_dim)

        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad = False

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc


    def forward(self, x, flag=False):
        x = self.convnet(x, flag)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]



    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

class Mask_IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        self.bn = nn.BatchNorm1d(self.feature_dim)

        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad = False

        self.aux_fc=None
        self.task_sizes=[]

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.feature_dim, new_task_size+1)


    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x, flag=False):
        x = self.convnet(x, flag)
        out = self.fc(x['features'])
        out["aux_logits"] = self.aux_fc(x['mask_features'])['logits']
        out.update(x)

        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

class Muti_Mask_Branch(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
    @property
    def feature_dim(self):
        return self.convnet.out_dim*len(self.convnet.mask_net.f_conv1_list)

    def update_fc(self, nb_classes):
        self.convnet.mask_net.append_network()
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.convnet.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x, flag=False):
        x = self.convnet(x, flag)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

class CosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())

class IncrementalNetWithBias(BaseNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
            out['logits'] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class Mask_IncrementalNetWithBias(IncrementalNetWithBias):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained, bias_correction)
        self.aux_fc=None

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out["aux_logits"] = self.aux_fc(x['mask_features'])['logits']
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
            out['logits'] = logits
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

        self.aux_fc = self.generate_fc(self.feature_dim, new_task_size + 1)

class DERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(DERNet,self).__init__()
        self.convnet_type=convnet_type
        self.convnets=nn.ModuleList()
        self.pretrained=pretrained
        self.out_dim=None
        self.fc=None
        self.aux_fc=None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out=self.fc(features)

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out.update({"aux_logits":aux_logits,"features":features})
        return out

    def update_fc(self, nb_classes):
        if len(self.convnets)==0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

class Mask_DERNet(DERNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)
        self.mask_aux_fc = None
    def forward(self, x):

        features = [convnet(x)['features'] for convnet in self.convnets[:-1]]
        outputs = self.convnets[-1](x)
        features.append(outputs['features'])
        features = torch.cat(features, 1)

        out=self.fc(features)

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out["mask_aux_logits"] = self.mask_aux_fc(outputs['mask_features'])['logits']

        out.update({"aux_logits":aux_logits,"features":features})
        return out

    def update_fc(self, nb_classes):
        if len(self.convnets)==0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size+1)
        self.mask_aux_fc = self.generate_fc(self.out_dim, new_task_size+1)

class SimpleCosineIncrementalNet(BaseNet):
    
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data=self.fc.sigma.data
            if nextperiod_initialization is not None:
                
                weight=torch.cat([weight,nextperiod_initialization])
            fc.weight=nn.Parameter(weight)
        del self.fc
        self.fc = fc
        

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

class AtteNet(BaseNet):
    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        self.bn = nn.BatchNorm1d(self.feature_dim)

        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad = False

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc
    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def backbone_learner(self):
        param = chain(self.convnet.conv1.parameters(), self.convnet.bn1.parameters(), self.convnet.layer1.parameters(),
                      self.convnet.layer2.parameters(),
                      self.convnet.layer3.parameters(),
                      self.convnet.layer4.parameters(),
                      self.fc.parameters()
                      )
        for p in param:
            yield p

    def attention_learner(self):
        param = chain(
                      self.convnet.f_conv1.parameters(),
                       self.convnet.f_conv2.parameters(),
                      self.convnet.f_conv3.parameters(),
                      self.convnet.f_conv4.parameters(),
                       )
        for p in param:
            yield p

    def reset_parameters(self, parameters):
        # reset fc  parameters
        for m in parameters:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x, flag=False):
        x = self.convnet(x, flag)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

class Sslnet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        self.projector_fc = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),)

        nf=64
        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.predictor = nn.Sequential(*layers)

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def reset_parameters(self, parameters):
        # reset fc  parameters
        for m in parameters:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convnet(x)
        feat = self.projector_fc(x['features'])
        out = self.fc(feat)
        # out['fc_features'] = feat
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out


    def forward_(self, x):
        x = self.convnet(x)
        feat = self.projector_fc(x['features'].detach())
        out = self.fc(feat)
        # out['fc_features'] = feat
        out.update(x)

        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)