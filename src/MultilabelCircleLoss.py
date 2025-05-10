from torch import nn as nn
import torch
class MultiLabelCircleLoss(nn.Module):
    def __init__(self):
        super(MultiLabelCircleLoss, self).__init__()
    def forward(self, Y_logit, Y):
        loss,cnt = 0,0
        for yp,yt in zip(Y_logit,Y):
            neg = yp[yt==0]
            pos = yp[yt==1]
            loss += torch.log(1+torch.exp(neg).sum()) + torch.log(1+torch.exp(-pos).sum())
            #loss += torch.log(1+(F.sigmoid(neg)**2*torch.exp(neg)).sum()) + torch.log(1+((1-F.sigmoid(pos))**2*torch.exp(-pos)).sum())
            #loss += len(yp) * (torch.log(1+torch.exp(neg).sum()/len(neg)) + torch.log(1+torch.exp(-pos).sum()/len(pos)))
            cnt += 1
        return loss/cnt
def MultiLabelCircleLoss_fun(Y_logit, Y):
    loss, cnt = 0, 0
    for yp, yt in zip(Y_logit, Y):
        neg = yp[yt == 0]
        pos = yp[yt == 1]
        loss += torch.log(1 + torch.exp(neg).sum()) + torch.log(1 + torch.exp(-pos).sum())
        # loss += torch.log(1+(F.sigmoid(neg)**2*torch.exp(neg)).sum()) + torch.log(1+((1-F.sigmoid(pos))**2*torch.exp(-pos)).sum())
        # loss += len(yp) * (torch.log(1+torch.exp(neg).sum()/len(neg)) + torch.log(1+torch.exp(-pos).sum()/len(pos)))
        cnt += 1
    return loss / cnt

def sparse_multilabel_categorical_crossentropy(label, pred, mask_zero=False, reduction='none'):
    """Sparse Multilabel Categorical CrossEntropy
     Reference: https://kexue.fm/archives/8888, https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
    Args:
     label: label tensor with shape [batch_size, n, num_positive] or [Batch_size, num_positive]
         should contain the indexes of the positive rather than a ont-hot vector
     pred: logits tensor with shape [batch_size, m, num_classes] or [batch_size, num_classes], don't use acivation.
     mask_zero: if label is used zero padding to align, please specify make_zero=True.
         when mask_zero = True, make sure the label start with 1 to num_classes, before zero padding.
    """
    zeros = torch.zeros_like(pred[..., :1])
    pred = torch.cat([pred, zeros], dim=-1)
    if mask_zero:
        infs = torch.ones_like(zeros) * float('inf')
        pred = torch.cat([infs, pred[..., 1:]], dim=-1)
    pos_2 = batch_gather(pred, label)
    pos_1 = torch.cat([pos_2, zeros], dim=-1)
    if mask_zero:
        pred = torch.cat([-infs, pred[..., 1:]], dim=-1)
        pos_2 = batch_gather(pred, label)
    pos_loss = torch.logsumexp(-pos_1, dim=-1)
    all_loss = torch.logsumexp(pred, dim=-1)
    aux_loss = torch.logsumexp(pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-16, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = pos_loss + neg_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))

def batch_gather(input, indices):
    """
    Args:
        input: label tensor with shape [batch_size, n, L] or [batch_size, L]
        indices: predict tensor with shape [batch_size, m, l] or [batch_size, l]
    Return:
        Note that when second dimention n != m, there will be a reshape operation to gather all value along this dimention of input
        if m == n, the return shape is [batch_size, m, l]
        if m != n, the return shape is [batch_size, n, l*m]
    """
    if indices.dtype != torch.int64:
        indices = torch.tensor(indices, dtype=torch.int64)
    results = []
    for data, indice in zip(input, indices):
        if len(indice) < len(data):
            indice = indice.reshape(-1)
            results.append(data[..., indice])
        else:
            indice_dim = indice.ndim
            results.append(torch.gather(data, dim=indice_dim-1, index=indice))
    return torch.stack(results)



