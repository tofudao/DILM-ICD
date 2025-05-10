import torch
from torch import nn

class Semble(nn.Module):
    def __init__(self,size,label_name, weight=None,bias=None):
        super(Semble, self).__init__()
        self.name = 'semble_{}'.format(label_name)
        # self.weight=nn.Parameter(weight)
        # self.bias = nn.Parameter(bias)
        self.pre_linears = nn.Linear(size, 1, bias=True)
        if weight is not None:
            print("初始化1")
            self.pre_linears.weight = nn.Parameter(weight)
            self.pre_linears.bias = nn.Parameter(bias)
        else:
            torch.nn.init.normal_(self.pre_linears.weight, 0, 0.03)
            self.pre_linears.bias.data.fill_(0)

    def forward(self,data):
        # torch.Size([16, 1, 768])
        # torch.Size([16, 50])
        # print(data.shape)
        logits=self.pre_linears.weight.mul(data).sum(dim=2).add(self.pre_linears.bias)
        # logits=self.weight.mul(data).sum(dim=2).add(self.bias)
        # print(logits.shape)
        return logits