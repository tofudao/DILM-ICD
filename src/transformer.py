from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict




class SelfAttention_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, name='selfAttn'):
        super(SelfAttention_PostLN, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(dropout)
        self.name = name
        #===================================================================
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
#         mean = 0.0
#         std = 0.03
#         torch.nn.init.normal(self.WQ.weight, mean, std)
#         self.WQ.bias.data.fill_(0)
#         torch.nn.init.normal(self.WK.weight, mean, std)
#         self.WK.bias.data.fill_(0)
#         torch.nn.init.normal(self.WV.weight, mean, std)
#         self.WV.bias.data.fill_(0)
#         torch.nn.init.normal(self.WO.weight, mean, std)
#         self.WO.bias.data.fill_(0)
        #===================================================================

    def forward(self, qx, kx, vx, maskPAD=None):
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        Bq,qL,C = qx.shape
        B,kvL,C = kx.shape
        # qx.shape.shape
        # torch.Size([1, 50, 768])
        # kx.shape
        # torch.Size([2, 2432, 768])
        # querie.shape
        # torch.Size([1, 1, 50, 128])
        # keys.shape
        # torch.Size([2, 1, 2432, 128])
        print("qx.shape",qx.shape)
        print("self.WQ",self.WQ)
        queries = self.WQ(qx).reshape(Bq,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × qL × dk
        keys    = self.WK(kx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × kvL × dk
#         values  = self.WV(vx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × kvL × dk
        
# #         queries = qx.reshape(Bq,qL,self.multiNum,self.dk).transpose(1,2)
# #         keys    = kx.reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × kvL × dk
# #         values  = vx.reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × kvL × dk
# #         queries = torch.randn(Bq,qL,self.multiNum,self.dk).transpose(1,2).to(device)
# #         keys    = torch.randn(B,kvL,self.multiNum,self.dk).transpose(1,2).to(device) # => batchSize × multiNum × kvL × dk
# #         values  = torch.randn(B,kvL,self.multiNum,self.dk).transpose(1,2).to(device) # => batchSize × multiNum × kvL × dk
# querie.shape torch.Size([1, 8, 8929, 1])
# keys.shape torch.Size([1, 8, 2223, 1])
# scores.shape torch.Size([1, 8, 8929, 2223])
# alpha.shape torch.Size([1, 8, 8929, 2223])
#scores.shape torch.Size([1, 8, 8929, 1244])
#         print("querie.shape",queries.shape)
#         print("keys.shape",keys.shape)
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × qL × kvL
# #         scores=torch.randn(B,self.multiNum,qL,kvL).to(device)
#         print("scores.shape",scores.shape)
#         if maskPAD is not None:
#             scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)
        #torch.Size([8, 1, 8929, n])
#         print(alpha.shape)
#         #====================
#         f=open(r'hello.txt','w')
#         print(alpha[:2,:,:2,:],file=f)
#         f.close()
        
#         print("alpha.shape",alpha.shape)

        # torch.Size([2, 1, 50, 128])
        z = self.dropout(alpha) @ keys # => batchSize × multiNum × qL × dk
        # torch.Size([2, 50, 128]
        z = z.transpose(1,2).reshape(B,qL,-1) # => batchSize × qL × multiNum*dk
        #z1 torch.Size([8, 50, 128])
        # print("z1",z.shape)
#         z=torch.randn(B,qL,C).to(device)
        z = self.WO(z) # => batchSize × qL × feaSize
        return z

class FFN_PostLN(nn.Module):
    def __init__(self, feaSize, dropout=0.1, actFunc=nn.GELU, name='FFN'):
        super(FFN_PostLN, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        actFunc(),
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        # ===================================================================
        nn.init.xavier_uniform_(self.Wffn[0].weight)
        nn.init.xavier_uniform_(self.Wffn[2].weight)
#         mean = 0.0
#         std = 0.03
#         torch.nn.init.normal(self.Wffn[0].weight, mean, std)
#         torch.nn.init.normal(self.Wffn[2].weight, mean, std)
#         self.Wffn[0].bias.data.fill_(0)
#         self.Wffn[2].bias.data.fill_(0)
        # ===================================================================
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × qL × feaSize
#         return z
#         ffnx = self.Wffn(z) # => batchSize × qL × feaSize

        return self.layerNorm2(z+self.dropout(z)) # => batchSize × qL × feaSize
    
class Transformer_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PostLN, self).__init__()
        self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
#         self.ffn = FFN_PreLN(feaSize, dropout)
        self.ffn = FFN_PostLN(feaSize, dropout)
    
    def forward(self, input):
        qx,kx,vx, maskPAD = input 
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        z = self.selfAttn(qx, kx, vx, maskPAD) # => batchSize × qL × feaSize
        return (self.ffn(qx, z), kx, vx, maskPAD)

class TransformerLayers_PostLN(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, dropout=0.1, name='textTransformer'):
        super(TransformerLayers_PostLN, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_PostLN(feaSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        # self.usePos = usePos
    def forward(self, qx, kx, vx, maskPAD=None):
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
#         qx=qx.repeat(2, 1, 1)
#         return (qx.repeat(8, 1, 1), kx,vx, maskPAD)
        return self.transformerLayers((qx,kx,vx, maskPAD)) # => batchSize × qL × feaSize



