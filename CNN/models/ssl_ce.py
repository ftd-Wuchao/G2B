import logging
import numpy as np
from PIL import Image
import math

import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.inc_net import IncrementalNet, Sslnet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# from losses import simclr_loss_func, SupConLoss, CenterLoss, CenterTripletLoss
from einops import repeat

output_dim = 128
lamda = 1
lamda_kd=2
T=2

init_epoch=200
init_lr=0.1
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005


# epochs = 80
# lrate = 0.1
# milestones = [40, 70]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4

epochs = 170
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay=2e-4

num_workers=8

# init_epoch=200
# init_lr=0.1
# init_milestones=[60,120,170]
# init_lr_decay=0.1
# init_weight_decay=0.0005
#
#
# epochs = 170
# lrate = 0.1
# milestones = [80, 120]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay=2e-4
# num_workers=8



class Ssl_ce(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = Sslnet(args['convnet_type'], False)
        self.ssl_loss = None
        self.init_loss(args)


        self.transforms =transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=32 // 20 * 2 + 1, sigma=(0.1, 2.0))], p=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),] )

        self.transforms_1 =transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),] )

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def init_loss(self,args):
        if args['ssl_loss'] == "BarlowTwins":
            self.ssl_loss = self.BarlowTwins
        elif args['ssl_loss'] == "SimSiam":
            self.ssl_loss = self.SimSiam
        elif args['ssl_loss'] == "BYOL":
            self.ssl_loss = self.BYOL

    def incremental_train(self, data_manager):
        # if self._cur_task<=0:
        self._cur_task += 1
        # self._cur_task=2
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes , self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())#
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

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

        if self._cur_task==0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch)) #init_epoch
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, image, targets) in enumerate(train_loader):
                image1, image2 = self.transformer_fn(image), self.transformer_fn(image)
                image1, image2, image = image1.to(self._device), image2.to(self._device), image.to(self._device)
                targets = targets.to(self._device)

                out1, out2 = self._network(image1), self._network(image2)
                feats1, feats2 = out1['features'], out2['features']
                logit1, logit2 = out1['logits'], out2['logits']

                logits = torch.cat([logit1, logit2], dim=0)
                targets = torch.cat([targets, targets], dim=0)
                loss_clf = F.cross_entropy(logits, targets)
                #
                loss_cons = self.ssl_loss(feats1, feats2)

                # feat = torch.cat([out1['fc_features'], out2['fc_features']], dim=0)
                # target = torch.cat([targets, targets], dim=0)
                # loss_center = self.cen_loss(feat, target)

                # logits = self._network(image)['logits']
                # loss_clf = F.cross_entropy(logits, targets)
                loss = loss_clf# + lamda*loss_cons #+ loss_center

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

        self.reset_parameters()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, image, targets) in enumerate(train_loader):
                image1, image2 = self.transformer_fn(image), self.transformer_fn(image)
                image1, image2, image = image1.to(self._device), image2.to(self._device), image.to(self._device)
                targets = targets.to(self._device)

                out1, out2 = self._network(image1), self._network(image2)
                feats1, feats2 = out1['features'], out2['features']
                logit1, logit2 = out1['logits'], out2['logits']

                logits = torch.cat([logit1, logit2], dim=0)
                #
                targets = torch.cat([targets, targets], dim=0)
                # fake_targets = targets - self._known_classes
                # loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                loss_clf=F.cross_entropy(logits,targets)

                loss_cons = self.ssl_loss(feats1, feats2)

                inputs = torch.cat((image1, image2),dim=0)
                loss_kd = _KD_loss(logits[:,:self._known_classes],self._old_network(inputs)["logits"], T)

                # feat = torch.cat([out1['fc_features'], out2['fc_features']], dim=0)
                # target = torch.cat([targets, targets], dim=0)
                # loss_center = self.cen_loss(feat, target)

                loss = loss_clf + lamda_kd*loss_kd #+ lamda*loss_cons
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

    def BarlowTwins(self, y1, y2):

        z1 = self._network.projector(y1)
        z2 = self._network.projector(y2)
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        c_diff = (c_ - torch.eye(D).cuda()).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()
        return loss

    def SimCLR(self, y1, y2, temp=100, eps=1e-6):
        z1 = self._network.projector(y1)
        z2 = self._network.projector(y2)
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)

        out = torch.cat([z_a, z_b], dim=0)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / temp)
        neg = sim.sum(dim=1)

        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temp)).cuda()
        neg = torch.clamp(neg - row_sub, min=eps)
        pos = torch.exp(torch.sum(z_a * z_b, dim=-1) / temp)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        return loss

    def SimSiam(self, y1, y2):
        def D(p, z):
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

        z1, z2 = self._network.projector(y1,), self._network.projector(y2)
        p1, p2 = self._network.predictor(z1), self._network.predictor(z2)

        loss = (D(p1, z2).mean() + D(p2, z1).mean()) * 0.5
        return loss

    def BYOL(self, y1, y2):
        def D(p, z):
            p = F.normalize(p, dim=-1, p=2)
            z = F.normalize(z, dim=-1, p=2)
            return 2 - 2 * (p * z).sum(dim=-1)

        z1, z2 = self._network.projector(y1, ), self._network.projector(y2)
        p1, p2 = self._network.predictor(z1), self._network.predictor(z2)

        loss = (D(z1, p2.detach()).mean() + D(z2, p1.detach()).mean()) * 0.5
        return loss

    def reset_parameters(self):
        # self._network.fc.weight.data.normal_(mean=0, std=0.01)
        # self._network.fc.bias.data.fill_(0.0)
        nn.init.kaiming_normal_(self._network.fc.weight, nonlinearity="linear")
        nn.init.constant_(self._network.fc.bias, 0.0)

        # for m in self._network.projector_fc.parameters():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
        #         nn.init.constant_(m.bias, 0.0)

                #nn.init.normal_(m.weight, std=0.01)

    def transformer_fn(self, x):
        bs = x.size()[0]
        ls = []
        # print(x.size())
        for i in range(bs):
            image = Image.fromarray(np.uint8(x[i]))
            ls.append(self.transforms_1(image).unsqueeze(0))
        x = torch.cat(ls,dim=0)
        return x

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum()/pred.shape[0]

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

        self.loss = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, target):

        out1, out2 = self.base_net(x1),  self.base_net(x2)
        feats1, feats2 = out1['features'], out2['features']
        logit1, logit2 = out1['logits'], out2['logits']

        f1 = F.normalize(self.projector(feats1), dim=1)
        f2 = F.normalize(self.projector(feats2), dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss(features, target)
        return loss, logit1, logit2
