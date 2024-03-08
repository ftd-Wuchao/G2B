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

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 80
lrate = 0.1
milestones = [40, 70]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8


class GMS(BaseLearner):

    def __init__(self, args, model=None):
        super().__init__(args)
        if model==None:
            self._network = IncrementalNet(args['convnet_type'], False)
        else:
            self._network = model
        self.weights = {n: p for n, p in self._network.named_parameters() if p.requires_grad}
        self.drop_ratio = args['ratio']
        self.mode = args['mode']
        self.update_type = args['update_ratio']
        # self._class_means = None

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
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=init_lr, weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones,
                                                       gamma=init_lr_decay)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):

            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()

                self.gradient_maximum_suppression()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, init_epoch, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def update_ratio(self, epoch, num_epoch):
        if "linear" in self.update_type:
            self.drop_ratio = epoch / num_epoch

    def calculate_global_mask(self):

        sizes = {}
        tensors = []
        all_params_size = 0
        if "element" in self.mode:
            for n, p in self.weights.items():
                if p.ndim == 4:
                    sizes[n] = p.grad.shape
                    tensors.append(torch.square(p.grad).view(-1))
                    all_params_size += torch.prod(torch.tensor(p.grad.shape)).item()

            tensors = torch.cat(tensors, 0)
            drop_num = int(all_params_size * self.drop_ratio) #- classifier_size

            top_pos = torch.topk(tensors, drop_num)[1]
            masks = torch.ones_like(tensors)
            masks[top_pos] = 0

            # masks = torch.zeros_like(tensors)
            # masks[top_pos] = 1

            mask_dict = {}
            now_idx = 0
            for k, v in sizes.items():
                end_idx = now_idx + torch.prod(torch.tensor(v))
                mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(self._device)
                now_idx = end_idx

        elif "block" in self.mode:
            for n, p in self.weights.items():
                if p.ndim == 4:
                    sizes[n] = p.grad.shape
                    tensors.append(torch.mean(torch.square(p.grad), [2,3]).view(-1))
                    all_params_size += sizes[n][0]*sizes[n][1]

            tensors = torch.cat(tensors, 0)
            drop_num = int(all_params_size * self.drop_ratio)

            top_pos = torch.topk(tensors, drop_num)[1]
            masks = torch.ones_like(tensors)
            masks[top_pos] = 0

            # masks = torch.zeros_like(tensors)
            # masks[top_pos] = 1

            mask_dict = {}
            now_idx = 0
            for k, v in sizes.items():
                end_idx = now_idx + v[0]*v[1]
                masks_ = masks[now_idx: end_idx].reshape(v[0]*v[1], 1, 1).expand(v[0]*v[1], v[2], v[3])
                mask_dict[k] = masks_.reshape(v).to(self._device)
                now_idx = end_idx
        else:
            assert False

        return mask_dict

    def calculate_local_mask(self):

        # element block

        mask_dict = {}
        for n, p in self.weights.items():
            if p.ndim == 4:
                if "element" in self.mode:
                    size= p.grad.shape

                    tensor = torch.square(p.grad).view(-1)

                    drop_num = int(torch.prod(torch.tensor(p.grad.shape)).item() * self.drop_ratio)

                    top_pos = torch.topk(tensor, drop_num)[1]

                    masks = torch.ones_like(tensor)
                    masks[top_pos] = 0

                    # masks = torch.zeros_like(tensor)
                    # masks[top_pos] = 1

                    mask_dict[n] = masks.reshape(size).to(self._device)

                elif "block" in self.mode:
                    size = p.grad.shape
                    tensor = torch.square(p.grad)
                    tensor_sum = torch.mean(tensor, dim=[2,3]).view(-1)

                    drop_num = int(size[0]*size[1] * self.drop_ratio)

                    top_pos = torch.topk(tensor_sum, drop_num)[1]

                    masks = torch.ones_like(tensor_sum)
                    masks[top_pos] = 0

                    # masks = torch.zeros_like(tensor_sum)
                    # masks[top_pos] = 1

                    masks = masks.reshape(size[0] * size[1], 1, 1).expand(size[0]*size[1], size[2], size[3])

                    mask_dict[n] = masks.reshape(size).to(self._device)
                else:
                    assert False

        return mask_dict

    def gradient_maximum_suppression(self):
        if 'global' in self.mode:
            masks = self.calculate_global_mask()
        elif 'local' in self.mode:
            masks = self.calculate_local_mask()
        for k, v in self.weights.items():
            if v.ndim==4:
                v.grad.data = v.grad.data * masks[k].data

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)
