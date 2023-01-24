from functools import total_ordering
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import os
from torch.nn import init
import sys
from utils import AverageMeter, accuracy, save_model
from torch.optim import lr_scheduler
import torch.optim as optim


class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps, h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps).forward(input)


class WSCNet(nn.Module):
    def __init__(self, model, num_classes=8, num_maps=4):
        super(WSCNet, self).__init__()
        self.model = model
        self.downconv = nn.Conv2d(2048, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.spatial_pooling1 = ClassWisePool(num_maps)
        self.spatial_pooling2 = ClassWisePool(num_classes)
        self.classifier = nn.Linear(4096, num_classes)
        # self.init_params()
    
    def forward(self, x):
        x = self.model(x)
        features = x
        x = self.downconv(x)
        x_conv = x
        x = self.GMP(x)
        x = self.spatial_pooling1(x)
        x = x.view(x.size(0), -1)

        x_conv = self.spatial_pooling1(x_conv)
        x_conv = x_conv * x.view(x.size(0), x.size(1), 1, 1)
        x_conv = self.spatial_pooling2(x_conv)
        x_conv_tmp = x_conv

        for i in range(2047):
            x_conv_tmp = torch.cat((x_conv_tmp, x_conv), 1)
        x_conv_tmp = torch.mul(x_conv_tmp, features)
        x_conv_tmp = torch.cat((features, x_conv_tmp), 1)
        x_conv_tmp = self.GAP(x_conv_tmp)
        x_conv_tmp = x_conv_tmp.view(x_conv_tmp.size(0), -1)
        res = self.classifier(x_conv_tmp)
        return x, res

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Trainer:
    def __init__(self, model, train_loader, test_loader, lr, args):
        super(Trainer, self).__init__()
        self.model = WSCNet(model).to(args.gpu)
        self.lr = lr
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.print_fn = print

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 20, gamma=0.1)

    def train(self, epoch):
        self.model.train()
        
        cls_losses = AverageMeter()
        aux_losses = AverageMeter()
        total_losses = AverageMeter()
        lr_last = 0

        for batch_idx, (train_data, target) in enumerate(self.train_loader):
            train_data, target = train_data.to(self.args.gpu), target.to(self.args.gpu)

            aux, logit = self.model(train_data)
            loss_aux = self.loss_fn(aux, target)
            loss_cls = self.loss_fn(logit, target)
            total_loss = 0.5 * loss_cls + 0.5 * loss_aux

            self.model.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            cls_losses.update(loss_cls.cpu().detach())
            aux_losses.update(loss_aux.cpu().detach())
            total_losses.update(total_loss.cpu().detach())
            
            lr_last = self.optimizer.param_groups[0]['lr']
        
        self.scheduler.step()
        self.print_fn("Epoch {}/{} train:last lr: {}, classification loss: {}, similarity loss: {}, total loss: {}".
                      format(epoch, self.args.epochs, lr_last, cls_losses.avg, aux_losses.avg, total_losses.avg))

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
            epoch, self.args.epochs, total_loss/total_num, top1, best_eval_acc
        ))

        if top1 == best_eval_acc:
            save_model('/home/ubuntu/jgl/projects/SOTA/ckp/WSCNet_FI_best.pth', self.model)
        
        return top1
