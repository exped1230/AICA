import numpy as np
import torch
from itertools import combinations


class TripletSelector(object):
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin1, margin2, emotions, dataset, negative_selection_fn, relation_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin1 = margin1
        self.margin2 = margin2
        self.emotions = emotions
        self.negative_selection_fn = negative_selection_fn
        self.relation_selection_fn = relation_selection_fn
        self.dataset = dataset

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        # if self.dataset == 'FI':
        #     emotions = [1, 0, 1, 1, 0, 1, 0, 0]
        # else:
        #     emotions = [0, 0, 0, 1, 1, 0, 0, 1]
        polaritys = []
        for cur_label in labels:
            polaritys.append(self.emotions[cur_label])
        triplets = []

        polaritys = np.array(polaritys)

        for label, polarity in zip(labels, polaritys):
            label_mask = (labels == label)
            polarity_mask = (polaritys == polarity)
            relation_mask = (labels != label) & polarity_mask
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
        # for polarity in set(polaritys):
        #     polarity_mask = (polaritys == polarity)
        #
        #     label_indices = np.where(label_mask)[0]
        #     if len(label_indices) < 2:
        #         continue

            negative_indices = np.where(np.logical_not(polarity_mask))[0]
            relation_indices = np.where(relation_mask)[0]

            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values1 = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(relation_indices)] + self.margin1
                loss_values1 = loss_values1.data.cpu().numpy()
                hard_relation = self.relation_selection_fn(loss_values1)

                if hard_relation is not None:
                    hard_relation = relation_indices[hard_relation]
                    loss_values2 = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(np.array([hard_relation]))] - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin2
                    loss_values2 = loss_values2.data.cpu().numpy()
                    hard_negative = self.negative_selection_fn(loss_values2)
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_relation, hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], relation_indices[0], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def semihard_relation(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def SemihardNegativeTripletSelector(margin1, margin2, emotion_polarity, dataset, cpu=False): 
    return FunctionNegativeTripletSelector(margin1=margin1, margin2=margin2, emotions=emotion_polarity,
        relation_selection_fn=lambda x: semihard_negative(x, margin1),
        negative_selection_fn=lambda x: semihard_relation(x, margin2), cpu=cpu, dataset=dataset)


class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
        topk: list of returned k
    
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    with torch.no_grad():
        maxk = max(topk)  # get k in top-k
        batch_size = target.size(0)  # get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)  # pred: [num of batch, k]
        pred = pred.t()  # pred: [k, num of batch]

        # [1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # np.shape(res): [k, 1]
        return res


from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''

    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def save_model(save_path, save_model):
    save_model.train()
    torch.save({'model': save_model.state_dict()}, save_path)
