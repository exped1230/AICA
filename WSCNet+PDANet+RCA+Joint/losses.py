import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):

    def __init__(self, margin1, margin2 , triplet_selector, args):
        super(OnlineTripletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.triplet_selector = triplet_selector
        self.args = args

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda(self.args.gpu)

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1) 
        ar_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 3]]).pow(2).sum(1)
        losses = F.relu(ap_distances - ar_distances + self.margin1)+F.relu(ar_distances - an_distances + self.margin2)

        return losses.mean(), len(triplets)