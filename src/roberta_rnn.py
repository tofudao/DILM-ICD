import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

###
from src.attention_layer import *
# from transformer import *
import pickle

# from src.MoE_transformer import *
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Roberta_rnn_model(RobertaPreTrainedModel):
    def __init__(self, config, args, vocab):
        super().__init__(config)
        self.name = "plm_rnn_model"
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # for name, param in self.roberta.named_parameters():
        #     if 'encoder.layer.11' not in name:
        #         param.requires_grad = False
        #     else:
        #         print(name)
        # for param in self.roberta.parameters():
        #     # print(param)
        #     param.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.attention = AttentionLayer(args=args, vocab=vocab)
        ###############################
        if "text_label" == args.attention_mode:
            # lab_file = '../data/mimic3/full/lable_title_full_768.pkl'
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

        self.hidden_size = args.hidden_size
        self.rnn = nn.LSTM(config.hidden_size, args.hidden_size, num_layers=1, batch_first=True, bidirectional=True,
                           dropout=0)

    # =============================================================================================================
    def layer_norm(self, a):
        for i in range(a.shape[0]):
            mean = a[i].mean()
            var = (a[i] - mean).pow(2).mean()
            a[i] = (a[i] - mean) / torch.sqrt(var + 1e-5)
        return a
    # =============================================================================================================
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(2, batch_size, self.hidden_size)).to(device)
        c = Variable(torch.zeros(2, batch_size, self.hidden_size)).to(device)
        return h, c

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            label_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()

        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)

        hidden = self.init_hidden(batch_size)
        rnn_output, hidden = self.rnn(hidden_output, hidden)

        hidden_output = self.dropout(rnn_output) + hidden_output

        logits = self.attention(x=hidden_output, label_batch=self.labDescVec)

        return logits
