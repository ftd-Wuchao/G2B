import logging
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 170
lrate = 0.1
milestones = [60, 100, 140]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class WA_new(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)

    def after_task(self):
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes - self._known_classes)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones,
                                                       gamma=init_lr_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def get_cur_index(self, targets):
        index = (targets >= self._known_classes) & (targets < self._total_classes)
        return index

    def get_recall(self, logits, targets, correct_list, total_list):
        targets_set = torch.unique(targets)
        for target in targets_set.tolist():
            target_index = target == targets

            logits_target = logits[target_index]
            targets_target = targets[target_index]
            _, preds = torch.max(logits_target, dim=1)
            correct_list[target] += preds.eq(targets_target.expand_as(preds)).cpu().sum()
            total_list[target] += len(targets_target)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch)) #init_epoch

        old_acc_list = []
        cur_acc_list = []
        recall_list = []
        loss_list = []
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            old_correct, old_total = 0, 0
            cur_correct, cur_total = 0, 0
            correct_list, total_list = np.zeros(self._total_classes), np.zeros(self._total_classes)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                self.get_recall(logits, targets, correct_list, total_list)

                cur_index = self.get_cur_index(targets)
                if torch.sum(cur_index) !=0:
                    _, cur_preds = torch.max(logits[cur_index], dim=1)
                    cur_correct += cur_preds.eq(targets[cur_index].expand_as(cur_preds)).cpu().sum()
                    cur_total += len(targets[cur_index])

                if torch.sum(~cur_index) !=0:
                    _, old_preds = torch.max(logits[~cur_index], dim=1)
                    old_correct += old_preds.eq(targets[~cur_index].expand_as(old_preds)).cpu().sum()
                    old_total += len(targets[~cur_index])

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if old_total!=0:
                old_acc_list.append(np.around(old_correct.item() * 100 / old_total, decimals=2))


            cur_acc_list.append(np.around(cur_correct.item() * 100 / cur_total, decimals=2))
            recall_list.append(np.mean(correct_list/total_list))
            loss_list.append(losses / len(train_loader))

            if epoch % 5 == 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
        # logging.info(info)
        logging.info('old_acc_list:{}'.format(old_acc_list))
        logging.info('cur_acc_list:{}'.format(cur_acc_list))
        logging.info('recall_list:{}'.format(recall_list))
        logging.info('loss_list:{}'.format(loss_list))

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))

        old_acc_list = []
        cur_acc_list = []
        recall_list = []
        loss_list = []
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            old_correct, old_total = 0, 0
            cur_correct, cur_total = 0, 0
            correct_list, total_list = np.zeros(self._total_classes), np.zeros(self._total_classes)
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(logits[:, :self._known_classes], self._old_network(inputs)["logits"], T)

                loss = loss_clf + 2 * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                self.get_recall(logits, targets, correct_list, total_list)

                cur_index = self.get_cur_index(targets)

                if torch.sum(cur_index) !=0:
                    cur_index = self.get_cur_index(targets)
                    _, cur_preds = torch.max(logits[cur_index], dim=1)
                    cur_correct += cur_preds.eq(targets[cur_index].expand_as(cur_preds)).cpu().sum()
                    cur_total += len(targets[cur_index])

                if torch.sum(~cur_index) !=0:
                    _, old_preds = torch.max(logits[~cur_index], dim=1)
                    old_correct += old_preds.eq(targets[~cur_index].expand_as(old_preds)).cpu().sum()
                    old_total += len(targets[~cur_index])

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if old_total!=0:
                old_acc_list.append(np.around(old_correct.item() * 100 / old_total, decimals=2))
            cur_acc_list.append(np.around(cur_correct.item() * 100 / cur_total, decimals=2))
            recall_list.append(np.mean(correct_list/total_list))
            loss_list.append(losses / len(train_loader))

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
        # logging.info(info)
        logging.info('old_acc_list:{}'.format(old_acc_list))
        logging.info('cur_acc_list:{}'.format(cur_acc_list))
        logging.info('recall_list:{}'.format(recall_list))
        logging.info('loss_list:{}'.format(loss_list))

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]