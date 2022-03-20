import torch.nn.functional as F
import torch


def CrossEntropy2d(predict, target, weight=None, reduction='mean'):
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
    assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
    assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))

    n, c, h, w = predict.size()

    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]

    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, weight=weight, reduction=reduction)

    return loss


def TripletMarginLoss(global_q, global_p, global_n, neg_counts, n_neg):
    criterion_vpr = torch.nn.TripletMarginLoss(margin=0.1 ** 0.5, p=2, reduction="sum")

    loss_triplet = 0
    for i, neg_count in enumerate(neg_counts):
        for n in range(neg_count):
            neg_index = (torch.sum(neg_counts[:i]) + n).item()
            loss_triplet += criterion_vpr(global_q[i:i + 1], global_p[i:i + 1], global_n[neg_index:neg_index + 1])

    loss_triplet /= n_neg.float().cuda()  # normalise by actual number of negatives

    return loss_triplet

