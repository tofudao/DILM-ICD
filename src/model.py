import torch
from torch import nn
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from src.embedding_layer import *
from src.attention_layer import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self,args,vocab):
        super(RNN, self).__init__()
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
        lstm_max_pool = self.lstm_max_pool(rnn_output).detach_()  # => (batchSize × 1 × filterSize)
        # logits=self.attention(x=rnn_output,label_batch=batch_data['labDescVec'],lstm_max_pool=lstm_max_pool)
        logits=self.attention(x=rnn_output,label_batch=self.labDescVec,max_pool=lstm_max_pool)

        return logits

