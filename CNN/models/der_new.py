import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet, Mask_DERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from losses.KD_loss import *

EPSILON = 1e-8

init_epoch=200
init_lr=0.1
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 170
lrate = 0.1
milestones = [80, 120,150]
lrate_decay = 0.1
batch_size = 128
weight_decay=2e-4 #0.0005
num_workers=8
T=2
#
# init_epoch=160
# init_lr=0.1
# init_milestones=[100,120]
# init_lr_decay=0.1
# init_weight_decay=0.0005
#
#
# epochs = 100
# lrate = 0.1
# milestones = [100,120]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=0.0005 #0.0005
# num_workers=8
# T=2


class DER(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = Mask_DERNet(args['convnet_type'], False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    def train(self):
        self._network.train()
        self._network.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)            
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs['logits']
                loss=F.cross_entropy(logits,targets)

                mask_aux_logits = outputs['mask_aux_logits']
                aux_targets = targets.clone()
                tensors_zeros = torch.zeros(aux_targets.size()).type_as(aux_targets).to(self._device)
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0,
                                          aux_targets - self._known_classes + 1, tensors_zeros)
                loss_mask_aux = F.cross_entropy(mask_aux_logits, aux_targets)

                loss += 0.5*loss_mask_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
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



    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                loss_clf=F.cross_entropy(logits,targets)

                aux_targets = targets.clone()
                tensors_zeros = torch.zeros(aux_targets.size()).type_as(aux_targets).to(self._device)
                aux_targets=torch.where(aux_targets-self._known_classes+1>0, aux_targets-self._known_classes+1, tensors_zeros)
                loss_aux=F.cross_entropy(aux_logits, aux_targets)

                mask_aux_logits = outputs['mask_aux_logits']
                loss_mask_aux = F.cross_entropy(mask_aux_logits, aux_targets)

                loss = loss_clf + loss_aux + 0.5*loss_mask_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)
        logging.info(info)