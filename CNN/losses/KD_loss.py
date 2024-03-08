
import math
import numpy as np
import torch
from torch.nn import functional as F

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]

def _MKD_loss(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="channels",
    normalize=True,
):

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)
        a = a ** 2
        b = b ** 2
        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "element":
            a = a.view(a.shape[0], -1)
            b = b.view(b.shape[0], -1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)


def _NMKD_loss(
    list_attentions_a,
    list_attentions_b,
    list_mask,
    collapse_channels="channels",
    normalize=True,
):

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b, c) in enumerate(zip(list_attentions_a, list_attentions_b, list_mask)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)
        a = a*c
        b = b*c
        a = a ** 2
        b = b ** 2
        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "element":
            a = a.view(a.shape[0], -1)
            b = b.view(b.shape[0], -1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)

def features_distillation8(
        list_attentions_a,
        list_attentions_b,
        nb_new_classes=1
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    # list_attentions_a = list_attentions_a[:-1]
    # list_attentions_b = list_attentions_b[:-1]
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        n, c, h, w = a.shape
        layer_loss = torch.tensor(0.).to(a.device)

        a = a ** 2
        b = b ** 2
        a_affinity_4 = F.avg_pool2d(a, (4, 4), stride=1, padding=2)
        b_affinity_4 = F.avg_pool2d(b, (4, 4), stride=1, padding=2)
        a_affinity_8 = F.avg_pool2d(a, (8, 8), stride=1, padding=4)
        b_affinity_8 = F.avg_pool2d(b, (8, 8), stride=1, padding=4)
        a_affinity_12 = F.avg_pool2d(a, (12, 12), stride=1, padding=6)
        b_affinity_12 = F.avg_pool2d(b, (12, 12), stride=1, padding=6)
        a_affinity_16 = F.avg_pool2d(a, (16, 16), stride=1, padding=8)
        b_affinity_16 = F.avg_pool2d(b, (16, 16), stride=1, padding=8)
        a_affinity_20 = F.avg_pool2d(a, (20, 20), stride=1, padding=10)
        b_affinity_20 = F.avg_pool2d(b, (20, 20), stride=1, padding=10)
        a_affinity_24 = F.avg_pool2d(a, (24, 24), stride=1, padding=12)
        b_affinity_24 = F.avg_pool2d(b, (24, 24), stride=1, padding=12)

        layer_loss = torch.frobenius_norm((a_affinity_4 - b_affinity_4).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_8 - b_affinity_8).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_16 - b_affinity_16).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_20 - b_affinity_20).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_24 - b_affinity_24).view(a.shape[0], -1), dim=-1).mean() + \
                     torch.frobenius_norm((a_affinity_12 - b_affinity_12).view(a.shape[0], -1), dim=-1).mean()
        layer_loss = layer_loss / 6.

        if i == len(list_attentions_a) - 1:
            # pod_factor = 0.0001 # cityscapes
            pod_factor = 0.0005  # voc
        else:
            # pod_factor = 0.0005 # cityscapes
            pod_factor = 0.01  # voc

        loss = loss + layer_loss.mean() #* 1.0 * pod_factor

    return loss / len(list_attentions_a)


def features_distillation_channel(
        list_attentions_a,
        list_attentions_b,
        nb_new_classes=1
):
    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    list_attentions_a = list_attentions_a[:-1]
    list_attentions_b = list_attentions_b[:-1]
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        n, c, h, w = a.shape
        layer_loss = torch.tensor(0.).to(a.device)

        a = a ** 2
        b = b ** 2

        # a_p = F.avg_pool2d(a.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))
        # b_p = F.avg_pool2d(b.permute(0, 2, 1, 3), (3, 1), stride=1, padding=(1, 0))

        a_p =  torch.nn.functional.adaptive_avg_pool2d(a, (1, 1)).squeeze()
        b_p =  torch.nn.functional.adaptive_avg_pool2d(b, (1, 1)).squeeze()

        layer_loss = torch.frobenius_norm((a_p - b_p).view(a.shape[0], -1), dim=-1).mean()

        if i == len(list_attentions_a) - 1:
            pod_factor = 0.0
        else:
            pod_factor = 0.0001  # cityscapes
        pod_factor = 0.01  # voc
        loss = loss + layer_loss.mean()# * pod_factor

    return loss / len(list_attentions_a)