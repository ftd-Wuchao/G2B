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

# from losses import simclr_loss_func, SupConLoss
from einops import repeat

temperature=0.1
output_dim = 128

init_epoch=500
init_lr=0.1
init_milestones=[300,400,450]
init_lr_decay=0.1
init_weight_decay=0.0005

epochs = 200
lrate = 0.1
milestones = [100,150]
lrate_decay = 0.1
batch_size = 1024 # 128
weight_decay=1e-4
num_workers=8

fc_epoch = 100
fc_lr = 1
fc_init_milestones=[60,75,90]
fc_init_lr_decay=0.2
fc_weight_decay= 0

class SimCLR(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.simclr_net = SimCLR_net(args)
        self._network = self.simclr_net.base_net

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

        if len(self._multiple_gpus) > 1:
            self.simclr_net = nn.DataParallel(self.simclr_net, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self.simclr_net = self.simclr_net.module

        # self._caculate_prototype(data_manager)

    def _train(self, train_loader, test_loader):
        self.simclr_net.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(self.simclr_net.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
            fc_optimizer = optim.SGD(self._network.fc.parameters(), lr=fc_lr,weight_decay=fc_weight_decay)
            fc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=fc_optimizer, milestones=fc_init_milestones, gamma=fc_init_lr_decay)
            self._init_train(train_loader,test_loader,optimizer,fc_optimizer, scheduler, fc_scheduler)
        else:
            optimizer = optim.SGD(self.simclr_net.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            fc_optimizer = optim.SGD(self._network.fc.parameters(), lr=fc_lr,weight_decay=fc_weight_decay)
            fc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=fc_optimizer, milestones=fc_init_milestones,
                                                          gamma=fc_init_lr_decay)
            self._update_representation(train_loader, test_loader, optimizer,fc_optimizer, scheduler, fc_scheduler)

    def _init_train(self,train_loader,test_loader,optimizer,fc_optimizer,scheduler, fc_scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.simclr_net.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                image1, image2 = inputs
                image1, image2 = image1.to(self._device), image2.to(self._device)
                targets = targets.to(self._device)

                loss = self.simclr_net(image1, image2, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()

            if epoch%5==0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader))
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader))
            prog_bar.set_description(info)

        logging.info(info)
        prog_bar = tqdm(range(fc_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                image1, image2 = inputs
                image1, image2 = image1.to(self._device), image2.to(self._device)
                targets = targets.to(self._device)

                with torch.no_grad():
                    x = self._network.convnet(image1)
                logits = self._network.fc(x['features'].detach())['logits']

                # x = self.simclr_net.forward_feat(image1)
                # logits = self._network.fc(x.detach())['logits']

                loss=F.cross_entropy(logits,targets)
                fc_optimizer.zero_grad()
                loss.backward()
                fc_optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            fc_scheduler.step()
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


    def _update_representation(self, train_loader, test_loader, optimizer, fc_optimizer, scheduler, fc_scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.simclr_net.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                image1, image2 = inputs
                image1, image2 = image1.to(self._device), image2.to(self._device)
                targets = targets.to(self._device)

                loss = self.simclr_net(image1, image2, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, '.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), )
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader))
            prog_bar.set_description(info)

        prog_bar = tqdm(range(fc_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                image1, image2 = inputs
                image1, image2 = image1.to(self._device), image2.to(self._device)
                targets = targets.to(self._device)

                with torch.no_grad():
                    x = self._network.convnet(image1)
                # x = self._network.convnet(image1)
                logits = self._network.fc(x['features'].detach())['logits']

                # x = self.simclr_net.forward_feat(image1)
                # logits = self._network.fc(x.detach())['logits']


                fake_targets=targets-self._known_classes
                loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets)

                loss=loss_clf

                fc_optimizer.zero_grad()
                loss.backward()
                fc_optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            fc_scheduler.step()
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


class SimCLR_net(nn.Module):
    def __init__(self, args):
        super(SimCLR_net, self).__init__()

        self.base_net = IncrementalNet(args['convnet_type'], False)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.base_net.feature_dim, self.base_net.feature_dim),
            nn.ReLU(),
            nn.Linear(self.base_net.feature_dim, output_dim),
        )

        # self.loss = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, target):
        feats1, feats2 = self.base_net(x1)['features'],  self.base_net(x2)['features']
        f1 = F.normalize(self.projector(feats1), dim=1)
        f2 = F.normalize(self.projector(feats2), dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss(features, target)
        return loss
