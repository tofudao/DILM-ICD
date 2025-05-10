import torch
from torch import nn
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from src.embedding_layer import *
from src.attention_layer import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN_Strim(nn.Module):
    def __init__(self,args,vocab):
        super(RNN_Strim, self).__init__()
        self.name="rnn_model"
        self.args = args
        self.vocab=vocab
        self.hidden_size = args.hidden_size
        self.output_size = self.hidden_size * 2

        self.embedding = EmbeddingLayer(embedding_mode=args.embedding_mode,
                                        pretrained_word_embeddings=vocab.word_embeddings,
                                        vocab_size=vocab.word_num)
        self.dropout = nn.Dropout(args.dropout)
        self.rnn = nn.LSTM(self.embedding.embedding_size, self.hidden_size, num_layers=1,bidirectional=True,dropout=0)
        self.relu = nn.ReLU()
        self.lstm_max_pool = nn.AdaptiveMaxPool2d((1, self.hidden_size * 2))
        # self.layer_norm = nn.LayerNorm([self.embedding.output_size])
        self.attention = AttentionLayer(args=args, vocab=vocab)

        ##################################################################
        # if args.is_trans:
        if True:
            # lab_file = '../data/mimic3/lable_title_768.pkl'
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


        ####################################################
        # token_len_strim=torch.nn.Parameter(init.normal_((torch.empty(10, 10), mean=0, std=1).unsqueeze(0),requires_grad=True)
        self.token_len_strim=torch.nn.Parameter(torch.randn(args.tok_stm,self.output_size, dtype=torch.float32).unsqueeze(0),
                                                requires_grad=True)
        self.label_strim=torch.nn.Parameter(torch.randn(args.label_stm,self.output_size, dtype=torch.float32).unsqueeze(0),
                                                requires_grad=True)

        self.transformer_label = TransformerLayers_PostLN(layersNum=1, feaSize=self.output_size, dk=args.dk,
                                                          multiNum=1,dropout=0.1)
        self.transformer_token = TransformerLayers_PostLN(layersNum=1, feaSize=self.output_size, dk=args.dk,
                                                          multiNum=1,dropout=0.1)

        self.transformer_interaction  = TransformerLayers_PostLN(layersNum=1, feaSize=self.output_size, dk=args.dk,
                                                          multiNum=args.multiNum,dropout=0.1)

        self.third_linears = nn.Linear(self.output_size,vocab.label_num, bias=True)
        self.third_linears = nn.Linear(args.dk,vocab.label_num, bias=True)


        self.label_strim=torch.nn.Parameter(torch.randn(args.label_stm,args.dk, dtype=torch.float32).unsqueeze(0),
                                                requires_grad=True)
        # self.W_la = nn.Linear(self.output_size, args.dk)
        self.W_os = nn.Linear(self.output_size, args.dk)
        # self.attns=nn.Linear(args.dk,args.label_stm)
        self.WO_rnn=nn.Linear(self.output_size, args.dk)
        self.WO=nn.Linear(args.dk, self.output_size)
        self.layerNorm1 = nn.LayerNorm([args.dk])
        self.layerNorm2 = nn.LayerNorm([self.output_size])

        self.dropout = nn.Dropout(0.1)

    #=============================================================================================================
    def layer_norm(self,a):
        for i in range(a.shape[0]):
            mean = a[i].mean()
            var = (a[i] - mean).pow(2).mean()
            a[i] = (a[i] - mean) / torch.sqrt(var + 1e-5)
        return a
    # =============================================================================================================
    def init_hidden(self,batch_size):
        h = Variable(torch.zeros(2, batch_size, self.hidden_size)).to(device)
        c = Variable(torch.zeros(2, batch_size, self.hidden_size)).to(device)
        return h, c
    def forward(self,batch_data):
        text_batch=batch_data['input_ids']
        lengths=batch_data['length_input'].to('cpu')

        batch_size = text_batch.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(text_batch)
        embeds = self.dropout(embeds)
        embeds = self.layer_norm(embeds)

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        rnn_output, hidden = self.rnn(embeds, hidden)
        # hidden = hidden[0]
        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)

        # lstm_max_pool = self.lstm_max_pool(rnn_output).detach_()  # => (batchSize × 1 × filterSize)
        # logits=self.attention(x=rnn_output,label_batch=batch_data['labDescVec'],lstm_max_pool=lstm_max_pool)
        # logits=self.attention(x=rnn_output,label_batch=self.labDescVec,max_pool=lstm_max_pool)
        # token_strim=self.transformer_token(qx=self.token_len_strim, kx=rnn_output, vx=rnn_output)
        # token_strim=token_strim[0]
        # label_strim=self.transformer_label(qx=self.labDescVec, kx=token_strim, vx=token_strim)
        # label_strim=label_strim[0]

        qx=self.label_strim
        kx=self.W_os(self.labDescVec)


        alpha = F.softmax(qx@kx.transpose(-1,-2), dim=-1)  # => batchSize × os × lable
        z = self.dropout(alpha) @ kx # => batchSize × os × dk
        # torch.Size([2, 50, 128]
        # z = self.WO(z)
        label_strim= self.layerNorm1(qx+self.dropout(z))  # => batchSize × os × inSize
        # label_strim=self.WO(label_strim)
        rnn_output = self.WO_rnn(rnn_output)


        att = self.transformer_interaction(qx=label_strim, kx=rnn_output, vx=rnn_output)
        att = att[0]
        att = torch.matmul(alpha.transpose(1, 2), att)  #=> batchSize × os × lable   batchSize × lable × inSize
        # att=self.layerNorm2(self.labDescVec+att)

        weighted_output = self.third_linears.weight.mul(att).sum(dim=2).add(self.third_linears.bias)
        logits = weighted_output.reshape(weighted_output.shape[0], -1)
        return logits

class DeepICDDescAttention(nn.Module):
    def __init__(self, inSize, classNum, labSize=1024, hdnDropout=0.1, attnList=[], labDescVec=None, name='DeepICDAttn'):
        super(DeepICDDescAttention, self).__init__()
        hdns,attns,bns = [],[],[]
        for i,os in enumerate(attnList):
            attns.append(nn.Linear(inSize,os))
            if i==len(attnList)-1:
                hdns.append(nn.Linear(inSize, labSize))
                inSize = labSize
            else:
                hdns.append(nn.Linear(inSize,inSize))
            bns.append(nn.BatchNorm1d(inSize))
        self.hdns = nn.ModuleList(hdns)
        self.attns = nn.ModuleList(attns)
        self.bns = nn.ModuleList(bns)
        self.dropout = nn.Dropout(p=hdnDropout)
        self.labDescVec = nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32)) if labDescVec is not None else None
        self.name = name
    def forward(self, X, labDescVec=None):
        if labDescVec is None:
            labDescVec = self.labDescVec
        # X: batchSize × seqLen × inSize
        for h,a,b in zip(self.hdns,self.attns,self.bns):
            alpha = F.softmax(a(X), dim=1) # => batchSize × seqLen × os
            X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × os × inSize
            X = h(X) # => batchSize × os × inSize
            X = b(X.transpose(1,2)).transpose(1,2) # => batchSize × os × inSize
            X = F.relu(X) # => batchSize × os × inSize
            X = self.dropout(X) # => batchSize × os × inSize
        # => batchSize × os × icdSize
        alpha = F.softmax(torch.matmul(X, labDescVec.transpose(0,1)), dim=1) # => batchSize × os × classNum
        X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × classNum × inSize
        return X