import logging
import numpy as np
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
from losses import  CenterLoss
from models.gms import GMS


# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,170]
# init_lr_decay=0.1
# init_weight_decay=0.0005
#
#
# epochs = 80
# lrate = 0.1
# milestones = [40, 70]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8

init_epoch=160
init_lr=0.1
init_milestones=[100,120]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 160
lrate = 0.1
milestones = [100,120]
lrate_decay = 0.1
batch_size = 128
weight_decay=0.0005
num_workers=8



class E_Finetune(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)
        self._class_means = None
        self.loss_function = CenterLoss(margin=0.1)  # TripletLoss CenterLoss CenterLoss(ap_margin=ap_margin, an_margin=an_margin)

    def after_task(self):

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

        fc_train_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='train',
                                                 mode='train')
        self.fc_train_loader = DataLoader(fc_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._caculate_prototype(data_manager)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)            
            self._init_train(train_loader,test_loader,optimizer,scheduler)

            # self.reset_parameters()
            optimizer = optim.SGD(self._network.fc.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
            # self.train_fc(self.fc_train_loader,optimizer,scheduler)

        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            # self._update_representation(train_loader, test_loader, optimizer, scheduler)

            self.reset_parameters()
            optimizer = optim.SGD(self._network.fc.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self.train_fc(self.fc_train_loader,optimizer,scheduler)

    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)['logits']
                loss_ce=F.cross_entropy(logits, targets)

                # features = self._network(inputs)['features']
                # loss_tri = self.loss_function(features, targets)[0]

                loss = loss_ce

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

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


            # info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
            # self._cur_task, epoch+1, init_epoch, losses/len(train_loader))

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
                logits = self._network(inputs)['logits']

                fake_targets=targets-self._known_classes
                loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets)

                # features = self._network(inputs)['features']
                # loss_tri = self.loss_function(features, targets)[0]

                loss = loss_clf

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
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def train_fc(self,train_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                feats = self._network.convnet(inputs)['features']
                logits = self._network.fc(feats.detach())['logits']
                loss_ce = F.cross_entropy(logits, targets)
                loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()


    def reset_parameters(self):
        # self._network.fc.weight.data.normal_(mean=0, std=0.01)
        # self._network.fc.bias.data.fill_(0.0)
        nn.init.kaiming_normal_(self._network.fc.weight, nonlinearity="linear")
        nn.init.constant_(self._network.fc.bias, 0.0)



