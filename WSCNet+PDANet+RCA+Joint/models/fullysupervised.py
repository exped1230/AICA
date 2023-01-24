from syslog import LOG_SYSLOG
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import AverageMeter, save_model
import math


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, test_loader, args):
        super(Trainer, self).__init__()
        self.model = model.to(args.gpu)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.print_fn = print

    def train(self, epoch):
        self.model.train()
        
        cls_losses = AverageMeter()
        lr_last = 0

        for batch_idx, (train_data, target) in enumerate(self.train_loader):
            train_data, target = train_data.cuda(self.args.gpu), target.cuda(self.args.gpu)
            
            logit = self.model(train_data)
            loss_cls = self.loss_fn(logit, target)

            self.model.zero_grad()
            loss_cls.backward()
            self.optimizer.step()

            cls_losses.update(loss_cls.cpu().detach())
            lr_last = self.optimizer.param_groups[0]['lr']

        self.scheduler.step()
        self.print_fn("Epoch {}/{} train: last lr: {}, classification loss: {}".
                      format(epoch, self.args.epochs, lr_last, cls_losses.avg))

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
        
        if top1 == best_eval_acc:
            save_model('/home/ubuntu12/jgl/projects/SOTA/ckp/base_best.pth', self.model)
        
        return top1
