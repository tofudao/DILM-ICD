import torch
from torch import nn
from transformer import *

class AttentionLayer(nn.Module):

    def __init__(self,args,config):

        super(AttentionLayer, self).__init__()

        # ===========================================
        self.layer = config.iter_layer
        self.attention_mode = config.attention_mode
        self.size = config.hidden_size*2
        self.d_a = config.d_a
        self.n_labels = config.label_num

        self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.third_linear = nn.Linear(config.hidden_size, config.num_labels)


        # =======================================================
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.reshape_fc = nn.Linear(self.size + self.n_labels, args.reshape_size, bias=True)

        self.att_fc = nn.Linear(args.reshape_size, self.size, bias=True)

        self.layerNorm1 = nn.LayerNorm([self.size])
        self.transformer_layer = TransformerLayers_PostLN(layersNum=1, feaSize=self.size, dk=args.dk, multiNum=args.multiNum,dropout=0.1)

        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03):


        torch.nn.init.normal_(self.first_linears.weight, mean, std)
        if self.first_linears.bias is not None:
            self.first_linears.bias.data.fill_(0)


        torch.nn.init.normal_(self.second_linears.weight, mean, std)
        if self.second_linears.bias is not None:
            self.second_linears.bias.data.fill_(0)

        torch.nn.init.normal_(self.third_linears.weight, mean, std)
        # =============================================================
        torch.nn.init.normal_(self.reshape_fc.weight, mean, std)
        self.reshape_fc.bias.data.fill_(0)
        torch.nn.init.normal_(self.att_fc.weight, mean, std)
        self.att_fc.bias.data.fill_(0)
        # ========================================================

    def forward(self, x , label_batch=None, lstm_max_pool=None):

        #label_batch.shape torch.Size([1, 50, 10])
        # x.shape torch.Size([8, 10, 10])
        if self.attention_mode == "text_label":
            att = self.transformer_layer(qx=label_batch, kx=x, vx=x)
            att = att[0]
            #att.shape torch.Size([8, 50, 10])
            # print("att.shape",att.shape)

            previous = None
            fcin = None
            for i in range(self.layer):
                if i == 0:
                    # print("第i层",i)
                    weighted_output = self.third_linears.weight.mul(att).sum(dim=2).add(self.third_linears.bias)
                else:
                    # print("第i层", i)
                    # att=att+att*previous
                    att = att * previous
                    att = self.layerNorm1(att)
                    # att=self.Tanh(att)
                    # att=att*previous
                    weighted_output = self.third_linears.weight.mul(att).sum(dim=2).add(self.third_linears.bias)
                out = weighted_output.unsqueeze(-1)
                if fcin is None:
                    fcin = out
                else:
                    fcin = fcin + out
                fea1 = fcin.permute(0, 2, 1).clone()  # =>  (batchSize × 1 × classNum)
                fea = fea1.detach_()
                # =================================================================================
                fea = torch.cat((lstm_max_pool, fea), 2)  # =>  (batchSize × 1 × (filterSize+classNum))/
                #             print("fea",fea)
                fea = self.reshape_fc(fea)  # =>  (batchSize × 1 × reshapesize)
                #             print("fea",fea)
                #             print("fea",fea.shape)
                # =================================================================================

                fc = self.att_fc(fea)  # =>  (batchSize × 1 × fea)
                #             print("fc",fc)
                previous = self.relu(fc)
        elif self.attention_mode == "laat":
            weights = F.tanh(self.first_linears(x))
            att_weights = self.second_linears(weights)
            # att_weights.shape torch.Size([1, 1440, 8929])
            # att_weights.shape torch.Size([1, 2337, 8929])
            print("att_weights.shape", att_weights.shape)
            att_weights = att_weights.transpose(1, 2)
            #  att_weights = F.softmax(att_weights, 1).transpose(1, 2)
            if len(att_weights.size()) != len(x.size()):
                att_weights = att_weights.squeeze()
            # 8*8929*2506,8*2506*1024，softmax是lenth维度，就是每一个标签的2506个不同分数加起来为1，表示每个词对这个标签的相关度，
            # 相乘后得到每一个标签的1024个特征，这些特征都是在这一特征维度词的分数加权和
            # att = att_weights @ x
            weighted_output = att_weights @ x
            weighted_output = self.third_linears.weight.mul(weighted_output).sum(dim=2).add(self.third_linears.bias)
        logits = weighted_output.reshape(fcin.shape[0], -1)

        return logits