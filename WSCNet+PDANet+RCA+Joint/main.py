import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
from dataset import MyDataset, BalancedBatchSampler
from resnet50 import resnet50
import torch.nn as nn
from utils import SemihardNegativeTripletSelector, get_cosine_schedule_with_warmup
import argparse
from losses import OnlineTripletLoss
import models.WSCNet as WSCNet, models.RCA as RCA, models.PDANet as PDANet, models.Joint as Joint, models.fullysupervised as fullysupervised


parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_dir', type=str, default='/home/ubuntu/jgl/datasets/Fi_old/by-image/train')
parser.add_argument('--test_dir', type=str, default='/home/ubuntu/jgl/datasets/Fi_old/by-image/test')
parser.add_argument('--dataset', type=str, default='FI')
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--idx_path', type=str, default='./sampled_label_idx_fi_1600.npy')
parser.add_argument('--alg', type=str, default='RCA')
parser.add_argument('--gpu', type=int, default=1)
args = parser.parse_args()

# idxes = np.load(args.idx_path)
train_dataset = MyDataset(args.train_dir, train=True)
test_dataset = MyDataset(args.test_dir, train=False)
print(len(train_dataset), len(test_dataset))

train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=8, n_samples=8)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=8, n_samples=16)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=8)

kwargs = {'num_workers': 4, 'pin_memory': True}

online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

model = resnet50(alg=args.alg, num_classes=8, pretrained=True, args=args)

lr = 1e-4
margin1 = 0.2
margin2 = 0.1

# 如果使用RCA方法，由于需要根据极性构造三元损失，因此需要根据数据集的类别提供每种情感的极性，下面是FI默认的情感极性
emotion_polarity = [1, 0, 1, 1, 0, 1, 0, 0]

loss_fn1 = OnlineTripletLoss(margin1, margin2, SemihardNegativeTripletSelector(margin1, margin2, emotion_polarity, args.dataset), args)
loss_fn2 = nn.CrossEntropyLoss()

if args.alg == 'RCA':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.epochs*len(train_loader))
    Trainer = RCA.Trainer(model, optimizer, scheduler, online_train_loader, online_test_loader, args=args)
    Trainer.set_loss_function(loss_fn1, loss_fn2)
elif args.alg == 'WSCNet':
    Trainer = WSCNet.Trainer(model, train_loader, test_loader, lr, args=args)
    Trainer.set_optimizer()
elif args.alg == 'PDANet':
    Trainer = PDANet.Trainer(model, train_loader, test_loader, lr, args=args)
    Trainer.set_optimizer()
elif args.alg == 'Joint':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    Trainer = Joint.Trainer(model, optimizer, scheduler, train_loader, test_loader, args=args)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    Trainer = fullysupervised.Trainer(model, optimizer, scheduler, train_loader, test_loader, args=args)

best_acc = 0

for i in range(args.epochs):
    Trainer.train(i+1)
    acc = Trainer.evaluate(i+1, best_acc)
    best_acc = max(best_acc, acc)
