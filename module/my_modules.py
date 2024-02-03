# -*- coding:utf-8 -*-

# @Filename: my_modules
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-06 15:34
# @Author  : Linshan
import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch
from audtorch.metrics.functional import pearsonr
import numpy as np

class CategoryAlign_Module(nn.Module):
    def __init__(self, num_classes=7, ignore_bg=False):
        super(CategoryAlign_Module, self).__init__()
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg

    def get_context(self, preds, feats):
        b, c, h, w = feats.size()
        _, num_cls, _, _ = preds.size()

        # softmax preds
        assert preds.max() <= 1 and preds.min() >= 0, print(preds.max(), preds.min())
        preds = preds.view(b, num_cls, 1, h * w)  # (b, num_cls, 1, hw)
        feats = feats.view(b, 1, c, h * w)  # (b, 1, c, hw)

        vectors = (feats * preds).sum(-1) / preds.sum(-1)  # (b, num_cls, C)

        if self.ignore_bg:
            vectors = vectors[:, 1:, :]  # ignore the background
        vectors = F.normalize(vectors, dim=1)
        return vectors

    def get_cross_corcoef_mat(self, preds1, feats1, preds2, feats2):
        context1 = self.get_context(preds1, feats1).mean(0)
        context2 = self.get_context(preds2, feats2).mean(0)

        n, c = context1.size()
        mat = torch.zeros([n, n]).to(context1.device)
        for i in range(n):
            for j in range(n):
                cor = pearsonr(context1[i, :], context2[j, :])
                mat[i, j] += cor[0]
        return mat

def Class_Distance_Compute(source, target, ignore_bg=False):
    """
    Compute cross-domain class distance
    """
    m = CategoryAlign_Module(ignore_bg=ignore_bg)
    S_preds, S_feats = source
    T_preds, T_feats = target
    S_preds, T_preds = S_preds.softmax(dim=1), T_preds.softmax(dim=1)

    cor_mat = m.get_cross_corcoef_mat(S_preds, S_feats,T_preds, T_feats).detach()
    cor_mat_numpy = cor_mat.cpu().numpy()
    diag = np.diag(cor_mat_numpy)
    return diag

def Global_Distance_Compute(feature_s, feature_t):
    kl_distance = nn.KLDivLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    distance = torch.sum(kl_distance(log_sm(feature_s), sm(feature_t)), dim=1)
    norm_distance = torch.exp(-distance)
    norm_distance = torch.mean(norm_distance[:]).detach()
    norm_distance = norm_distance.cpu().numpy()
    return norm_distance

if __name__ == '__main__':
    num_classes = 7
    mat = torch.ones([num_classes, num_classes]) * -1
    print(mat.shape)
    diag = torch.diag_embed(torch.Tensor([2]).repeat(1, num_classes))
    print(diag)
    print(mat + diag)

