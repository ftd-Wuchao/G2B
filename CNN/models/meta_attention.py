import logging
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, AtteNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

T=2
lamda=3

#resnet18

batch_size = 256
num_workers=8

inner_step = 1

init_epoch=200
init_milestones=[60,120,170]
init_inner_lr=0.01
init_inner_lr_decay=0.5
init_inner_weight_decay=0.0005
init_outer_lr=0.01
init_outer_lr_decay=0.0
init_outer_weight_decay=0.0005


epochs = 80
milestones = [40, 70]
inner_lr = 0.01
inner_lr_decay = 0.5
inner_weight_decay=2e-4
outer_lr = 0.01
outer_lr_decay = 0.0
outer_weight_decay=2e-4



class Meta_attention(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = AtteNet(args['convnet_type'], False)

        self.weights = {n: p for n, p in self._network.named_parameters() if p.requires_grad}

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._caculate_prototype(data_manager)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            optimizer_backbone = optim.SGD(self._network.backbone_learner(), momentum=0.9,lr=init_inner_lr,weight_decay=init_inner_weight_decay)
            scheduler_backbone = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_backbone, milestones=init_milestones, gamma=init_inner_lr_decay)
            optimizer_attention = optim.SGD(self._network.attention_learner(), momentum=0.9,lr=init_outer_lr,weight_decay=init_outer_weight_decay)
            scheduler_attention = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_attention, milestones=init_milestones, gamma=init_outer_lr_decay)
            self._init_train(train_loader, test_loader, optimizer_backbone, scheduler_backbone, optimizer_attention, scheduler_attention)
        else:
            optimizer_backbone = optim.SGD(self._network.backbone_learner(), momentum=0.9,lr=inner_lr,weight_decay=inner_weight_decay)
            scheduler_backbone = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_backbone, milestones=milestones, gamma=inner_lr_decay)
            optimizer_attention = optim.SGD(self._network.attention_learner(), momentum=0.9,lr=outer_lr,weight_decay=outer_weight_decay)
            scheduler_attention = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_attention, milestones=milestones, gamma=outer_lr_decay)
            self._update_representation(train_loader, test_loader, optimizer_backbone, scheduler_backbone, optimizer_attention, scheduler_attention)

    def _init_train(self, train_loader, test_loader, optimizer_backbone, scheduler_backbone, optimizer_attention, scheduler_attention):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # backbone_data, backbone_label, atte_data, atte_label = self.get_data(inputs, targets, 0.5)
                # backbone_data, backbone_label, atte_data, atte_label = \
                #     backbone_data.to(self._device), backbone_label.to(self._device) , atte_data.to(self._device), atte_label.to(self._device)
                backbone_data, backbone_label = inputs.to(self._device), targets.to(self._device)
                # self._network.reset_parameters(self._network.attention_learner())

                # self._network.reset_parameters(self._network.attention_learner())
                # net = self._network.copy()
                # lr = optimizer_backbone.state_dict()['param_groups'][0]['lr']
                # optimizer_net = optim.SGD(net.backbone_learner(), momentum=0.9, lr=lr,
                #                                weight_decay=init_inner_weight_decay)
                # for _ in range(inner_step):
                logits =  self._network(backbone_data)['logits']
                loss = F.cross_entropy(logits, backbone_label)
                # net.backbone_learner() = self.update_params()
                optimizer_backbone.zero_grad()
                # optimizer_attention.zero_grad()
                loss.backward()
                optimizer_backbone.step()
                # optimizer_attention.step()
                #
                # logits = net(atte_data)['logits']
                # loss = F.cross_entropy(logits, atte_label)
                # self.update_params(loss, net, lr)
                # for _ in range(inner_step):
                #     logits =  self._network(backbone_data)['logits']
                #     loss = F.cross_entropy(logits, backbone_label)
                #     # net.backbone_learner() = self.update_params()
                #     optimizer_backbone.zero_grad()
                #     loss.backward()
                #     optimizer_backbone.step()
                #
                # logits = self._network(atte_data)['logits']
                # loss = F.cross_entropy(logits, atte_label)
                # optimizer_attention.zero_grad()
                # loss.backward()
                # optimizer_attention.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(backbone_label.expand_as(preds)).cpu().sum()
                total += len(backbone_label)
                # del net
            scheduler_backbone.step()
            scheduler_attention.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            if epoch%5==0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def _update_representation(self, train_loader, test_loader, optimizer_backbone, scheduler_backbone, optimizer_attention, scheduler_attention):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # backbone_data, backbone_label, atte_data, atte_label = self.get_data(inputs, targets, 0.5)
                # backbone_data, backbone_label, atte_data, atte_label = \
                #     backbone_data.to(self._device), backbone_label.to(self._device) , atte_data.to(self._device), atte_label.to(self._device)
                backbone_data, backbone_label = inputs.to(self._device), targets.to(self._device)
                # self._network.reset_parameters(self._network.attention_learner())

                logits = self._network(backbone_data)['logits']
                fake_targets = backbone_label - self._known_classes
                loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                loss_kd = _KD_loss(logits[:, :self._known_classes], self._old_network(atte_data)["logits"], T)
                loss = lamda * loss_kd + loss
                optimizer_backbone.zero_grad()
                # optimizer_attention.zero_grad()
                loss.backward()
                optimizer_backbone.step()
                # optimizer_attention.step()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(backbone_label.expand_as(preds)).cpu().sum()
                total += len(backbone_label)

            scheduler_backbone.step()
            scheduler_attention.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def get_data(self, data, label, ratio = 0.8, tactics = 'meta'):
        batch_size = label.size()[0]

        # print(uiq_label)
        if tactics=='random':
            train_index = torch.zeros(batch_size)
            train_index[:int(batch_size*ratio)] = 1
            train_index = train_index==1
        elif tactics=='meta':
            uiq_label, index = torch.unique(label, sorted=False, return_inverse=True)
            boud = int(len(uiq_label)*ratio)

            train_index = torch.zeros(batch_size)
            train_index = train_index==1
            for i in range(boud):
                train_index = train_index | (index == i)
        backbone_data = data[train_index] # backbone_data backbone_label atte_data atte_label
        backbone_label = label[train_index]

        atte_data = data[~train_index]
        atte_label = label[~train_index]

        return backbone_data, backbone_label, atte_data, atte_label

    def update_params(self, loss, net, lr):

        create_graph = True
        net_weights = {n: p for n, p in self._network.named_parameters() if p.requires_grad}
        grads = torch.autograd.grad(loss, net_weights.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(self._network.named_parameters(), grads):
            if grad is not None:
                self.weights[name].data = param - lr * grad



def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]