import logging
import numpy as np
import math
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from losses.KD_loss import _KD_loss, _MKD_loss,_NMKD_loss,features_distillation8,features_distillation_channel
from losses.Sparisty_loss import  Sparisty_loss, IntraClass_loss
# init_lr=0.1
# lrate = 0.1
#
#
# init_epoch=200
# init_milestones=[60,120,160]
# init_lr_decay=0.1
# init_weight_decay=0.0005
#
#
# epochs = 250
# milestones = [60,120, 180,220]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8

init_epoch=200
init_lr=0.01
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005

epochs = 80
lrate = 0.01
milestones = [40, 70]
lrate_decay = 0.1
batch_size = 128
weight_decay=2e-4
num_workers=8

T=2
lamda=3
lamda_sparisty=0.5
lamda_intraclass=0.25

class LwF_new(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):

        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(self._total_classes / (self._total_classes - self._known_classes))

        # logging.info('Adaptive factor: {}'.format(self.factor))

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task==0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)            
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            loss_sparisty = 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)
                logits = out['logits']

                loss_sparisty = Sparisty_loss(out['mask'])
                loss_intraclass =  IntraClass_loss(out['mask'], targets)

                loss=F.cross_entropy(logits,targets) + lamda_sparisty* loss_sparisty #+ lamda_intraclass*loss_intraclass #
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
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Sparisty_Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), loss_sparisty, train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)
                logits = out['logits']

                fake_targets=targets-self._known_classes
                loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets)
                loss_kd =_KD_loss(logits[:,:self._known_classes],self._old_network(inputs)["logits"],T)

                loss_sparisty = Sparisty_loss(out['mask'])

                # mask = out['mask']
                # neg_mask = out['neg_mask']
                # loss_kd_mask = features_distillation_channel(mask, self._old_network(inputs)["mask"])
                # #loss_kd_mask = features_distillation8(mask, self._old_network(inputs)["fmaps"])
                # #loss_kd_mask = _MKD_loss(mask, self._old_network(inputs)["fmaps"], collapse_channels="channels")  # _MKD_loss  features_distillation8
                # #loss_kd_mask = _NMKD_loss(mask, self._old_network(inputs)["mask"], neg_mask, collapse_channels="gap") # _MKD_loss  features_distillation8

                loss_intraclass = IntraClass_loss(out['mask'], targets)

                loss = lamda*loss_kd + loss_clf + lamda_sparisty* loss_sparisty# + lamda_intraclass*loss_intraclass #+ lamda_sparisty* loss_sparisty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Sparisty_Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader),loss_sparisty, train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

