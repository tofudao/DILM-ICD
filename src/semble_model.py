# import torch
# from torch import nn

# class Semble(nn.Module):
#     def __init__(self,size,name):
#         super(Semble, self).__init__()
#         self.name=name
#         self.pre_linears =nn.Linear(size,1, bias=True)
#         torch.nn.init.normal_(self.pre_linears.weight, 0, 0.03)
#         self.pre_linears.bias.data.fill_(0)

#     def forward(self,data):

#         # print("semble:")
#         # print("data",data.shape)
#         # print("self.pre_linears.weight",self.pre_linears.weight.shape)

#         logits=self.pre_linears.weight.mul(data).sum(dim=2).add(self.pre_linears.bias)
#         # print("logits:",logits.shape)
#         return logits

import torch
from torch import nn

class Semble(nn.Module):
    def __init__(self,size,label_name, weight=None,bias=None):
        super(Semble, self).__init__()
        self.name = 'semble_{}'.format(label_name)
        self.pre_linears = nn.Linear(size, 1, bias=True)
        if weight is not None:
            print("初始化1")
            self.pre_linears.weight = nn.Parameter(weight)
            self.pre_linears.bias = nn.Parameter(bias)
        else:
            torch.nn.init.normal_(self.pre_linears.weight, 0, 0.03)
            self.pre_linears.bias.data.fill_(0)

    def forward(self,data):
        logits=self.pre_linears.weight.mul(data).sum(dim=2).add(self.pre_linears.bias)

        return logits

# import torch
# from torch import nn

#
# class Semble(nn.Module):
#     def __init__(self, size, label_name, model_state):
#         super(Semble, self).__init__()
#         self.pre_linears = nn.Linear(size, 1, bias=True)
#         i = label_name
#         for names, param in model_state.named_parameters():
#             if 'attention.third_linears.weight' not in names and 'attention.third_linears.bias' not in names:
#                 param.requires_grad = False
#                 # print(" ",type(param)," ",names)
#             elif 'attention.third_linears.weight' in names:
#                 param = nn.Parameter(param[i, :])
#                 self.pre_linears.weight = nn.Parameter(param)
#             elif 'attention.third_linears.bias' in names:
#                 param = nn.Parameter(param[i])
#                 self.pre_linears.bias = nn.Parameter(param)
#
#         self.name = 'semble_{}'.format(label_name)
#         self.model_state = model_state
#
#     def forward(self, data):
#         # for names, param in self.model_state.named_parameters():
#         #     if '.attention' not in names:
#         #         print(" ",names," ",param)
#         # print(data.shape)
#         # print(self.pre_linears.weight.shape)
#         logits = self.pre_linears.weight.mul(data).sum(dim=2).add(self.pre_linears.bias)
#         # print(logits.shape)
#         # logits=self.model_state(data)
#
#         return logits


