from syslog import LOG_SYSLOG
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import AverageMeter
import math


# set_loss_function, set_margin
class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, test_loader, args):
        super(Trainer, self).__init__()
        self.model = model.to(args.gpu)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.loss_fn1 = nn.CrossEntropyLoss()
        self.loss_fn2 = nn.KLDivLoss(reduction='batchmean')
        self.emotions = torch.Tensor([0, 7, 2, 1, 6, 3, 4, 5]).to(args.gpu)
        self.polarity = torch.Tensor([1, 0, 1, 1, 0, 1, 0, 0]).to(args.gpu)
        self.print_fn = print

    def generate_distribution(self, targets):
        batch_size = targets.size(0)
        target_distribution = torch.zeros((batch_size, 8))
        C = math.sqrt(2*math.pi)
        for i in range(batch_size):
            cur_target = targets[i]
            target_emotion = self.emotions[cur_target]
            for idx, cur_emotion in enumerate(self.emotions):
                if self.polarity[idx] != self.polarity[cur_target]:
                    target_distribution[i][idx] = 0
                else:
                    target_distribution[i][idx] = (torch.exp(-(target_emotion-cur_emotion)**2/2) / C) + (0.1 / 8)
        target_distribution = target_distribution / target_distribution.sum(1, keepdim=True)
        return target_distribution

    def train(self, epoch):
        self.model.train()
        
        cls_losses = AverageMeter()
        dis_losses = AverageMeter()
        total_losses = AverageMeter()
        lr_last = 0

        for batch_idx, (train_data, target) in enumerate(self.train_loader):
            train_data, target = train_data.cuda(self.args.gpu), target.cuda(self.args.gpu)
            cls_target = target
            dis_target = self.generate_distribution(target).to(self.args.gpu)

            logit = self.model(train_data)
            loss_cls = self.loss_fn1(logit, cls_target)
            loss_dis = self.loss_fn2(torch.log_softmax(logit, dim=1), dis_target)
            total_loss = 0.8 * loss_dis + 0.2 * loss_cls

            self.model.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            cls_losses.update(loss_cls.cpu().detach())
            dis_losses.update(loss_dis.cpu().detach())
            total_losses.update(total_loss.cpu().detach())
            lr_last = self.optimizer.param_groups[0]['lr']

        self.scheduler.step()
        self.print_fn("Epoch {}/{} train: last lr: {}, classification loss: {}, distribution loss: {}, total loss: {}".
                      format(epoch, self.args.epochs, lr_last, cls_losses.avg, dis_losses.avg, total_losses.avg))

    @torch.no_grad()
    def evaluate(self, epoch, best_acc):
        self.model.eval()
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        for batch_idx, (x, y) in enumerate(self.test_loader):
            x, y = x.cuda(self.args.gpu), y.cuda(self.args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            prediction = self.model(x)
            loss = F.cross_entropy(prediction, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(prediction, dim=-1)[1].cpu().detach().tolist())
            total_loss += loss.cpu().detach() * num_batch
        top1 = (np.array(y_true) == np.array(y_pred)).sum() / total_num

        best_eval_acc = max(best_acc, top1)
        self.print_fn("Epoch {}/{} test: test loss: {}, top-1 acc: {}, best top-1 acc: {}".format(
            epoch, self.args.epochs, total_loss/total_num, top1, best_eval_acc))
        
        return top1
