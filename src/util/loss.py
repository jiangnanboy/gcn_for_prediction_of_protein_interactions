import torch
import torch.nn.functional as F

def vgae_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    '''
    变分图自编码，损失函数包括两部分：
        1.生成图和原始图之间的距离度量
        2.节点表示向量分布和正态分布的KL散度
    '''
    # 负样本边的weight都为1，正样本边的weight都为pos_weight
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def gae_loss_function(preds, labels, norm, pos_weight):
    '''
    图自编码，损失函数是生成图和原始图之间的距离度量
    '''
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost