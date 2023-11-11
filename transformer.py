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
        nn.init.xavier_uniform(self.WQ.weight.weight)
        nn.init.xavier_uniform(self.WK.weight.weight)
        nn.init.xavier_uniform(self.WV.weight.weight)
        nn.init.xavier_uniform(self.WO.weight.weight)

        # mean = 0.0
        # std = 0.03
        # torch.nn.init.normal(self.WQ.weight, mean, std)
        # self.WQ.bias.data.fill_(0)
        # torch.nn.init.normal(self.WK.weight, mean, std)
        # self.WK.bias.data.fill_(0)
        # torch.nn.init.normal(self.WV.weight, mean, std)
        # self.WV.bias.data.fill_(0)
        # torch.nn.init.normal(self.WO.weight, mean, std)
        # self.WO.bias.data.fill_(0)
        #===================================================================

    def forward(self, qx, kx, vx, maskPAD=None):
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        B,qL,C = qx.shape
        kvL = kx.shape[1]
        queries = self.WQ(qx).reshape(B,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × qL × dk
        keys    = self.WK(kx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × kvL × dk
        values  = self.WV(vx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × kvL × dk

        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × qL × kvL

        if maskPAD is not None:
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)

        z = self.dropout(alpha) @ values # => batchSize × multiNum × qL × dk
        z = z.transpose(1,2).reshape(B,qL,-1) # => batchSize × qL × multiNum*dk

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
        # ===================================================================
        # mean = 0.0
        # std = 0.03
        nn.init.xavier_uniform(self.Wffn[0].weight)
        nn.init.xavier_uniform(self.Wffn[2].weight)
        # torch.nn.init.normal(self.Wffn[0].weight, mean, std)
        # torch.nn.init.normal(self.Wffn[2].weight, mean, std)
        # self.Wffn[0].bias.data.fill_(0)
        # self.Wffn[2].bias.data.fill_(0)
        # ===================================================================

        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × qL × feaSize
        ffnx = self.Wffn(z) # => batchSize × qL × feaSize

        return self.layerNorm2(z+self.dropout(ffnx)) # => batchSize × qL × feaSize
    
class Transformer_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PostLN, self).__init__()
        self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum)
        # self.ffn = FFN_PreLN(feaSize, dropout)
        self.ffn = FFN_PostLN(feaSize, dropout)

    def forward(self, input):
        qx,kx,vx, maskPAD = input 
        # qx,kx,vx: batchSize × qL,kvL,kvL × feaSize; maskPAD: batchSize × qL × kvL
        z = self.selfAttn(qx, kx, vx, maskPAD) # => batchSize × qL × feaSize
        return (self.ffn(qx, z), kx,vx, maskPAD) # => batchSize × qL × feaSize
        # return (self.ffn(qx, z), kx, vx, maskPAD)

class TransformerLayers_PostLN(nn.Module):
    def __init__(self, seqMaxLen, layersNum, feaSize, dk, multiNum, maxItems=10, dropout=0.1, name='textTransformer'):
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
        return self.transformerLayers((qx,kx,vx, maskPAD)) # => batchSize × qL × feaSize