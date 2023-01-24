import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import os
from torch.nn import init
import sys
from utils import AverageMeter, accuracy
from torch.optim import lr_scheduler
import torch.optim as optim


class spatial_block(nn.Module):
    def __init__(self, num_channels):
        super(spatial_block, self).__init__()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(num_channels, num_channels)
        self.conv = nn.Conv2d(num_channels, num_channels, 1)
        self.conv2 = nn.Conv2d(num_channels, 1, 1)

    def forward(self, x, channel_wise):
        out = self.conv(x)
        if len(channel_wise.size()) != len(x.size()):
            channel_wise = self.fc(channel_wise)
            channel_wise = channel_wise.view(-1, channel_wise.size(1), 1, 1)
            out = out+channel_wise
        out = self.tanh(out)
        out = self.conv2(out)
        x_shape = out.size(2)
        y_shape = out.size(3)
        out = out.view(-1, x_shape*y_shape)
        out = F.softmax(out, dim=1)
        out = out.view(-1, 1, x_shape, y_shape)
        out = x*out
        out = torch.mean(out.view(-1, out.size(1), out.size(2)*out.size(3)), 2)
        return out


class SENet_block(nn.Module):
    def __init__(self, num_channels):
        super(SENet_block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_classify = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self,x):
        out1 = self.conv1(x)
        out1 = self.sigmoid(out1)
        out1 = torch.mean(out1.view(-1, out1.size(1), out1.size(2)*out1.size(3)), 2)
        out_channel_wise = out1
        out2 = self.conv2(x)
        out2 = self.relu(out2)
        out = out2*out_channel_wise.view(-1, out1.size(1), 1, 1)
        return out, out_channel_wise, out2


class PDANet(nn.Module):
    def __init__(self, model, num_classes=8):
        super(PDANet, self).__init__()
        self.base = model
        num_channels = 2048
        self.se_block = SENet_block(num_channels)
        # self.fc = nn.Linear(num_channels*2, 3)
        self.fc_classify = nn.Linear(num_channels*2, num_classes)
        self.spatial = spatial_block(num_channels)

    def forward(self, x):
        x = self.base(x)
        out, out_channel_wise, out2 = self.se_block(x)
        out = torch.mean(out.view(-1, out.size(1), out.size(2)*out.size(3)), 2)
        spatial_feature = self.spatial(out2, out_channel_wise)
        feature_cat = torch.cat((out, spatial_feature), 1)
        # out = self.fc(feature_cat)
        out_classify = self.fc_classify(feature_cat)
        return out_classify


class Trainer:
    def __init__(self, model, train_loader, test_loader, lr, args):
        super(Trainer, self).__init__()
        self.model = PDANet(model).to(args.gpu)
        self.lr = lr
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.print_fn = print

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 100, gamma=0.1)

    def train(self, epoch):
        self.model.train()
        
        cls_losses = AverageMeter()
        lr_last = 0

        for batch_idx, (train_data, target) in enumerate(self.train_loader):
            train_data, target = train_data.to(self.args.gpu), target.to(self.args.gpu)

            logit = self.model(train_data)
            loss = self.loss_fn(logit, target)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            cls_losses.update(loss.cpu().detach())
            
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
            epoch, self.args.epochs, total_loss/total_num, top1, best_eval_acc
        ))
        
        return top1
