from syslog import LOG_SYSLOG
from imageio import save
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import AverageMeter, accuracy, save_model


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
        self.print_fn = print

    def set_loss_function(self, loss_fn1, loss_fn2):
        self.loss_fn1 = loss_fn1
        self.loss_weight1 = 0.1
        self.loss_fn2 = loss_fn2
        self.loss_weight2 = 0.9

    def train(self, epoch):
        self.model.train()
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        cls_losses = AverageMeter()
        sim_losses = AverageMeter()
        total_losses = AverageMeter()
        lr_last = 0
        batch_data_time = AverageMeter()
        batch_model_time = AverageMeter()

        start_batch.record()

        for batch_idx, (train_data, target) in enumerate(self.train_loader):
            train_data, target = train_data.cuda(self.args.gpu), target.cuda(self.args.gpu)
            end_batch.record()
            torch.cuda.synchronize()
            batch_data_time.update(start_batch.elapsed_time(end_batch) / 1000)
            start_run.record()

            feature, prediction = self.model(train_data)
            loss_sim, _ = self.loss_fn1(feature, target)
            loss_cls = self.loss_fn2(prediction, target)
            total_loss = self.loss_weight1 * loss_sim + self.loss_weight2 * loss_cls

            self.model.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            cls_losses.update(loss_cls.cpu().detach())
            sim_losses.update(loss_sim.cpu().detach())
            total_losses.update(total_loss.cpu().detach())
            
            lr_last = self.optimizer.param_groups[0]['lr']

            end_run.record()
            torch.cuda.synchronize()
            batch_model_time.update(start_run.elapsed_time(end_run) / 1000)

            start_batch.record()
        self.scheduler.step()
        self.print_fn("Epoch {}/{} train: data time: {}, model time: {}, last lr: {}, classification loss: {}, similarity loss: {}, total loss: {}".
                      format(epoch, self.args.epochs, batch_data_time.avg, batch_model_time.avg, lr_last, cls_losses.avg, sim_losses.avg, total_losses.avg))

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
            _, prediction = self.model(x)
            loss = F.cross_entropy(prediction, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(prediction, dim=-1)[1].cpu().detach().tolist())
            total_loss += loss.cpu().detach() * num_batch
        top1 = (np.array(y_true) == np.array(y_pred)).sum() / total_num

        best_eval_acc = max(best_acc, top1)
        self.print_fn("Epoch {}/{} test: test loss: {}, top-1 acc: {}, best top-1 acc: {}".format(
            epoch, self.args.epochs, total_loss/total_num, top1, best_eval_acc))
        
        if top1 == best_eval_acc:
            save_model('/home/ubuntu12/jgl/projects/SOTA/ckp/RCA_FI_best.pth', self.model)

        return top1
