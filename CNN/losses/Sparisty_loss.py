import math
import numpy as np
import torch
from torch.nn import functional as F

EPS=1e-8

def Sparisty_loss(list_mask, normalize='L2', shrinkingfn='exponential', lfs_loss_fn_touse='lasso'):
    loss = torch.tensor(0.).to(list_mask[0].device)
    for mask in list_mask:
        # shape of (b, n, w, h)
        channel_mask = torch.flatten(torch.nn.AdaptiveAvgPool2d((1, 1))(mask), 1)
        if normalize == 'L1':
            channel_mask_norm = F.normalize(channel_mask, p=1, dim=1)
        elif normalize == 'L2':
            channel_mask_norm = F.normalize(channel_mask, p=2, dim=1)
        elif normalize == 'max_foreachfeature':
            channel_mask_norm = channel_mask / (torch.max(channel_mask, dim=1, keepdim=True).values + EPS)
        elif normalize == 'max_maskedforclass':
            channel_mask_norm = torch.zeros_like(channel_mask)
            classes = torch.unique(channel_mask_norm)
            if classes[-1] == 0:
                classes = classes[:-1]
            for cl in classes:
                cl_mask = channel_mask_norm == cl
                channel_mask_norm += (channel_mask / (torch.max(
                    channel_mask[cl_mask.expand(-1, channel_mask.shape[1], -1, -1)]) + EPS)) * cl_mask.float()
        elif normalize == 'max_overall':
            channel_mask_norm = channel_mask / (torch.max(channel_mask) + EPS)
        elif normalize == 'softmax':
            channel_mask_norm = torch.softmax(channel_mask, dim=1)
        # single_loss = torch.mean(abs(channel_mask_norm))

        if shrinkingfn == 'squared':
            shrinked_value = torch.sum(channel_mask_norm**2, dim=1, keepdim=True)
        if shrinkingfn == 'power3':
            shrinked_value = torch.sum(channel_mask_norm ** 3, dim=1, keepdim=True)
        elif shrinkingfn == 'exponential':
            shrinked_value = torch.sum(torch.exp(channel_mask_norm), dim=1, keepdim=True)

        summed_value = torch.sum(channel_mask_norm, dim=1, keepdim=True)

        if lfs_loss_fn_touse == 'ratio':
            single_loss = shrinked_value / (summed_value + EPS)
        elif lfs_loss_fn_touse == 'lasso':  # NB: works at features space directly
            single_loss = torch.norm(channel_mask_norm,1)/ channel_mask_norm.numel() # simple L1 (Lasso) regularization
        elif lfs_loss_fn_touse == 'max_minus_ratio':
            # TODO: other loss functions to be considered
            # outputs = summed_value - shrinked_value / summed_value
            pass
        elif lfs_loss_fn_touse == 'entropy':  # NB: works only with probabilities (i.e. with L1 or softmax as normalization)
            single_loss = torch.sum(- channel_mask_norm * torch.log(channel_mask_norm + 1e-10), dim=1)
        loss += single_loss.mean()
    return loss / len(list_mask)

def IntraClass_loss(list_mask, labels, normalize='L2'):
    loss = torch.tensor(0.).to(list_mask[0].device)
    label_set = torch.unique(labels)
    for mask in list_mask:
        channel_mask = torch.flatten(torch.nn.AdaptiveAvgPool2d((1, 1))(mask), 1)
        if normalize == 'L2':
            channel_mask = F.normalize(channel_mask, p=2, dim=1)
        elif normalize == 'max_foreachfeature':
            channel_mask = channel_mask / (torch.max(channel_mask, dim=1, keepdim=True).values + EPS)
        class_loss = torch.tensor(0.).to(list_mask[0].device)
        for label in label_set.tolist():
            index = label == labels
            class_mask = channel_mask[index]
            class_mask_mean = torch.mean(class_mask.detach(), dim=0)
            # print(class_mask.size(), class_mask_mean.size())
            single_loss = torch.mean(class_mask-class_mask_mean)
            class_loss += single_loss
        loss += class_loss/len(label_set)
    return loss / len(list_mask)


# # Features Sparsification loss defined in SDR: https://arxiv.org/abs/2103.06342
# class FeaturesSparsificationLoss(nn.Module):
#     def __init__(self, lfs_normalization, lfs_shrinkingfn, lfs_loss_fn_touse, mask=False, reduction='mean'):
#         super().__init__()
#         self.mask = mask
#         self.lfs_normalization = lfs_normalization
#         self.lfs_shrinkingfn = lfs_shrinkingfn
#         self.lfs_loss_fn_touse = lfs_loss_fn_touse
#         self.eps = 1e-15
#         self.reduction = reduction
#
#     def forward(self, features, labels, val=False):
#         outputs = torch.tensor(0.)
#
#         if not val:
#             labels = labels.unsqueeze(dim=1)
#             labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()
#
#             if self.lfs_normalization == 'L1':
#                 features_norm = F.normalize(features, p=1, dim=1)
#             elif self.lfs_normalization == 'L2':
#                 features_norm = F.normalize(features, p=2, dim=1)
#             elif self.lfs_normalization == 'max_foreachfeature':
#                 features_norm = features / (torch.max(features, dim=1, keepdim=True).values + self.eps)
#             elif self.lfs_normalization == 'max_maskedforclass':
#                 features_norm = torch.zeros_like(features)
#                 classes = torch.unique(labels_down)
#                 if classes[-1] == 0:
#                     classes = classes[:-1]
#                 for cl in classes:
#                     cl_mask = labels_down == cl
#                     features_norm += (features / (torch.max(features[cl_mask.expand(-1, features.shape[1], -1, -1)]) + self.eps)) * cl_mask.float()
#             elif self.lfs_normalization == 'max_overall':
#                 features_norm = features / (torch.max(features) + self.eps)
#             elif self.lfs_normalization == 'softmax':
#                 features_norm = torch.softmax(features, dim=1)
#
#             if features_norm.sum() > 0:
#                 if self.lfs_shrinkingfn == 'squared':
#                     shrinked_value = torch.sum(features_norm**2, dim=1, keepdim=True)
#                 if self.lfs_shrinkingfn == 'power3':
#                     shrinked_value = torch.sum(features_norm ** 3, dim=1, keepdim=True)
#                 elif self.lfs_shrinkingfn == 'exponential':
#                     shrinked_value = torch.sum(torch.exp(features_norm), dim=1, keepdim=True)
#
#                 summed_value = torch.sum(features_norm, dim=1, keepdim=True)
#
#                 if self.lfs_loss_fn_touse == 'ratio':
#                     outputs = shrinked_value / (summed_value + self.eps)
#                 elif self.lfs_loss_fn_touse == 'lasso':  # NB: works at features space directly
#                     outputs = torch.norm(features, 1) / 2  # simple L1 (Lasso) regularization
#                 elif self.lfs_loss_fn_touse == 'max_minus_ratio':
#                     # TODO: other loss functions to be considered
#                     # outputs = summed_value - shrinked_value / summed_value
#                     pass
#                 elif self.lfs_loss_fn_touse == 'entropy':  # NB: works only with probabilities (i.e. with L1 or softmax as normalization)
#                     outputs = torch.sum(- features_norm * torch.log(features_norm + 1e-10), dim=1)
#
#         if self.reduction == 'mean':
#             return outputs.mean()
#         elif self.reduction == 'sum':
#             return outputs.sum()
