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
from models.gms import GMS

init_epoch=200
init_lr=0.1
init_milestones=[60,120,160]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 250
lrate = 0.1
milestones = [60,120, 180,220]
lrate_decay = 0.1
batch_size = 128
weight_decay=2e-4
num_workers=8
T=2
lamda=3

class E_LwF(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)

        self.use_gms = args['USE_GMS']
        if self.use_gms:
            self.gms = GMS(args, model=self._network)
        self.task_class_nums = []

    def after_task(self):

        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_class_nums.append(self._total_classes-self._known_classes)
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

        self._caculate_prototype(data_manager)


    def _train(self, train_loader, test_loader):
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

        minimizer = ASAM(optimizer, self._network)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs)
                logits = output['logits']
                features = output['features']

                loss=F.cross_entropy(logits,targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()


                # # Ascent Step
                # predictions = self._network(inputs)['logits']
                # batch_loss = F.cross_entropy(predictions, targets)
                # batch_loss.mean().backward()
                # minimizer.ascent_step()
                #
                # # Descent Step
                # logits = self._network(inputs)['logits']
                # loss = F.cross_entropy(logits, targets).mean()
                # loss.backward()
                # minimizer.descent_step()

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
        minimizer = ASAM(optimizer, self._network)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fake_targets = targets - self._known_classes

                output = self._network(inputs)
                logits = output['logits']
                features = output['features']
                loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets)
                loss_kd =_KD_loss(logits[:, :self._known_classes], self._old_network(inputs)["logits"], T)

                loss=lamda*loss_kd+loss_clf
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                # # Ascent Step
                # logits = self._network(inputs)['logits']
                # loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets).mean()
                # loss_kd =_KD_loss(logits[:, :self._known_classes], self._old_network(inputs)["logits"], T)
                # loss=lamda*loss_kd+loss_clf
                # loss.backward()
                # minimizer.ascent_step()
                #
                # # Descent Step
                # logits = self._network(inputs)['logits']
                # loss_clf = F.cross_entropy(logits[:,self._known_classes:], fake_targets).mean()
                # loss_kd =_KD_loss(logits[:, :self._known_classes], self._old_network(inputs)["logits"], T)
                # loss=lamda*loss_kd+loss_clf
                # loss.backward()
                # minimizer.descent_step()

                losses += loss.item()

                with torch.no_grad():
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
            # gamma = epoch/epochs
            # self.weight_average(gamma)
        logging.info(info)

    # def get_old_class_feature(self, current_task, batch_size):
    #
    #     for task_id in range(current_task):
    #         samples_num = batch_size/self.task_class_nums[self._cur_task]
    #         for i in range(self.task_class_nums[task_id]):
    #             sampler = torch.distributions.multivariate_normal.MultivariateNormal(self._train_class_mean[task_id, i],
    #                                                                            self._covariance_matrix[task_id, i])
    #             samples = sampler.sample()
    #
    # def weight_average(self, gamma):
    #     for param_t, param in zip(self._network.convnet.parameters(), self._old_network.convnet.parameters()):
    #         param_t.data.mul_(gamma).add_(1 - gamma, param.data)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]

def _features_KD_loss(old_features, new_features):
    return 0.5*torch.sum((new_features - old_features).pow(2))/new_features.shape[0]

from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()



