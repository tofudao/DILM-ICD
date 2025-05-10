import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import pickle
from src.embedding_layer import *
from src.attention_layer import *
#这个变量可能与循环神经网络（Recurrent Neural Networks, RNNs）的更新机制有关。如果设置为 1，
#可能意味着在 RNN 中使用不同的隐藏状态更新机制。如果设置为 0，则可能表示使用标准的或默认的更新机制
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM=0

class Mamba(nn.Module):
    def __init__(self, args,vocab):
        super(Mamba, self).__init__()
        seq_len=args.max_seq_length
        self.embedding = EmbeddingLayer(embedding_mode=args.embedding_mode,
                                        pretrained_word_embeddings=vocab.word_embeddings,
                                        vocab_size=vocab.word_num)
        d_model = args.embedding_size
        state_size = args.state_size
        batch_size=args.batch_size
        self.mamba_block = MambaBlock(seq_len, d_model, state_size,batch_size)

        self.dropout = nn.Dropout(args.dropout)
        self.lstm_max_pool = nn.AdaptiveMaxPool2d((1, d_model))
        self.attention = AttentionLayer(args=args, vocab=vocab)
        ##################################################################
        if args.is_trans:
            lab_file = args.label_titlefile
            with open(lab_file, 'rb') as file:
                labDescVec = pickle.load(file)
                # type(labDescVec) <class 'dict'>
                print("labDescVec的key", labDescVec.keys())
                labDescVec = labDescVec['id2descVec']
                # label_to_id_full=labDescVec['label_to_id']
            self.labDescVec = torch.nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32).unsqueeze(0),
                                            requires_grad=True)
            # labDescVec = torch.nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32).unsqueeze(0))
        else:
            self.labDescVec = None
        ########################################################################
    # =============================================================================================================
    def layer_norm(self, a):
        for i in range(a.shape[0]):
            mean = a[i].mean()
            var = (a[i] - mean).pow(2).mean()
            a[i] = (a[i] - mean) / torch.sqrt(var + 1e-5)
        return a
    # =============================================================================================================

    def forward(self, batch_data):
        text_batch = batch_data['input_ids']
        embeds = self.embedding(text_batch)
        embeds = self.dropout(embeds)
        embeds = self.layer_norm(embeds)
        mamba = self.mamba_block(embeds)
        lstm_max_pool = self.lstm_max_pool(mamba).detach_()  # => (batchSize × 1 × filterSize)
        logits=self.attention(x=mamba,label_batch=self.labDescVec,max_pool=lstm_max_pool)

        return logits

class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size,batch_size):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, state_size)
        self.fc3 = nn.Linear(d_model, state_size)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)

        # h  [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model)

    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2  MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:

            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True

                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,
                                                                                                               "b l d -> b l d 1") * self.dB

            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y  [batch_size, seq_len, d_model]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y

        else:
            # h [batch_size, seq_len, d_model, state_size]
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # y  [batch_size, seq_len, d_model]
            y = torch.einsum('bln,bldn->bld', self.C, h)

            return y


class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, batch_size):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)

        # For residual skip connection
        self.D = nn.Linear(d_model, 2 * d_model)

        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2 * d_model, state_size, batch_size)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1)

        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model)

        # rmsnorm
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
        # Refer to Figure 3 in the MAMBA paper

        x = self.norm(x)

        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)

        x_conv_act = F.silu(x_conv)

        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)

        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))

        x_combined = x_act * x_residual

        x_out = self.out_proj(x_combined)

        return x_out


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


