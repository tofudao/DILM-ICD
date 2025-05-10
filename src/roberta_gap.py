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

from src.MoE_transformer import *
from torch.autograd import Variable

# from src.attention_layer2 import *
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Roberta_model(RobertaPreTrainedModel):
    def __init__(self, config, args, vocab):
        super().__init__(config)
        self.name = "plm_model"
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # for name, param in self.roberta.named_parameters():
        #     if 'encoder.layer' in name and 'encoder.layer.11' not in name:
        #         param.requires_grad = False
        #         print("10和11",name)
        #     else:
        #         print(name)
        # for param in self.roberta.parameters():
        #     # print(param)
        #     param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.dropout)  # Dropout(p=0.1, inplace=False)
        self.init_weights()

        self.plm_maxpool = nn.AdaptiveMaxPool2d((1, config.hidden_size))
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

        # ======================================================================
        # self.transformer_layer = TransformerLayers_PostLN(layersNum=1, feaSize=config.hidden_size, dk=args.dk,
        #                                                   multiNum=args.multiNum, dropout=0.1)
        self.transformer_layer = Transformer_Block(feaSize=config.hidden_size, dk=args.dk, multiNum=args.multiNum)
        # self.chunk_maxpool = nn.AdaptiveMaxPool2d((1, config.hidden_size))
        # self.chunk_meanpool = nn.AdaptiveAvgPool2d((1, self.output_size))
        self.output_size = config.hidden_size
        self.chunk_size = args.chunk_size
        self.num_chunks = int(args.max_seq_length / self.chunk_size)
        self.chunk_meanpool = nn.AdaptiveAvgPool2d((1, self.output_size))
        self.pe = self.positional_encoding(max_sequence_length=50, d_model=config.hidden_size)
        ########################################################################
        self.hidden_size = args.hidden_size
        self.rnn = nn.LSTM(config.hidden_size, args.hidden_size, num_layers=1, batch_first=True, bidirectional=True,
                           dropout=0.1)
        # self.moe_layer = SwitchFeedForward(capacity_factor=1, drop_tokens=False, is_scale_prob=False,
        #                                    n_experts=args.num_experts, d_model=args.hidden_size * 2, expert_dim=args.dk)

        # self.moe_layer =SwitchMoE(dim=args.hidden_size * 2,hidden_dim=args.dk,num_experts=args.num_experts,topk_expert=args.topk, use_aux_loss=True)
        self.moe_layer = SwitchMoE(dim=args.hidden_size * 2, hidden_dim=args.expert_dk, num_experts=args.num_experts,
                                   topk_expert=args.topk)

    # =============================================================================================================
    def layer_norm(self, a):
        for i in range(a.shape[0]):
            mean = a[i].mean()
            var = (a[i] - mean).pow(2).mean()
            a[i] = (a[i] - mean) / torch.sqrt(var + 1e-5)
        return a

    def positional_encoding(self, max_sequence_length, d_model):

        pe = torch.zeros(max_sequence_length, d_model).to(device)
        position = torch.arange(0, max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    # def get_positional_encoding(self, seq_len, emb_dim, positionwise=False):
    #     """Generate positional encoding."""
    #     position_enc = torch.zeros(seq_len, emb_dim)
    #     for pos in range(seq_len):
    #         for i in range(emb_dim):
    #             if positionwise:
    #                 position_enc[pos, i] = torch.pow(10000, (2 * (i // 2) / emb_dim))
    #             else:
    #                 position_enc[pos, i] = torch.pow(10000, (2 * ((i % 2) + pos) / emb_dim))
    #     return position_enc
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
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()
        # batch_size, num_chunks, chunk_size  torch.Size([2, 5, 2])
        # print("batch_size, num_chunks, chunk_size ",input_ids.size())
        # input_ids.view(-1, chunk_size) torch.Size([10, 2])
        # print("input_ids.view(-1, chunk_size)",input_ids.view(-1, chunk_size).size())
        # batch_size, num_chunks, chunk_size
        # torch.Size([1, 5, 2])
        # input_ids.view(-1, chunk_size)
        # torch.Size([5, 2])
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
        #===============================================================
        gap_list = kwargs.get('gap_list', None)
        batch_size, num_chunks, chunk_size = gap_list.size()
        print("batch_size",batch_size.shape)
        print("batch_size",num_chunks.shape)
        print("batch_size",chunk_size.shape)
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

        #==========================================================

        # print(outputs)
        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)

        # print(hidden_output.shape)
        # BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[ grad_fn=<NativeLayerNormBackward0>), pooler_output=None, hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
        # 结果3 torch.Size([12, 128, 768])
        # torch.Size([19, 128, 768])
        # print("结果3",outputs[0].size())#torch.Size([5, 2, 768])
        ###############################
        # MOE
        # #====================================================================
        # hidden_moe_output=self.moe_layer(hidden_output)
        # #====================================================================
        # hidden_output=self.dropout(hidden_moe_output)+hidden_output
        # #====================================================================

        # chunk = outputs[0].view(batch_size, num_chunks , chunk_size,-1)
        # cls_token_hidden_state = outputs.last_hidden_state[:, 0, :]

        # #==========================================================================
        # # #===============================================================
        # chunk = outputs[0].view(batch_size, num_chunks , chunk_size,-1)
        # chunk_att = self.chunk_meanpool(chunk)
        # chunk_att=torch.squeeze(chunk_att,dim=2)
        # #=======================================================================
        # #位置编码
        # # pos=self.get_positional_encoding(seq_len=num_chunks,emb_dim=self.output_size, positionwise=True)
        # chunk_att=chunk_att + self.pe[:, :chunk_att.size(1)]

        # #===================================================================
        # # selfatt
        # hidden_output=self.transformer_layer(qx=chunk_att, kx=chunk_att, vx=chunk_att)
        # # =============================================================================
        # # rnn
        # hidden = self.init_hidden(batch_size)
        # rnn_output, hidden = self.rnn(chunk_att, hidden)
        # # hidden_output=torch.unsqueeze(rnn_output,dim=2)

        # # =============================================================================

        # hidden_output=torch.unsqueeze(hidden_output[0],dim=2)
        # hidden_output= (chunk+hidden_output).view(batch_size,num_chunks * chunk_size , self.output_size)
        # # # #==================================================================
        # rnn
        #
        # hidden = self.init_hidden(batch_size)
        # rnn_output, hidden = self.rnn(hidden_output, hidden)
        # hidden_output = rnn_output+self.dropout(hidden_moe_output)

        # #================================================================
        # hidden_output=self.moe_layer(hidden_output)
        # ===========================================================
        maxpool = self.plm_maxpool(outputs[0].view(batch_size, num_chunks, -1)).detach_()

        # logits, moeloss = self.attention(x=hidden_output, label_batch=self.labDescVec, max_pool=maxpool)
        logits = self.attention(x=hidden_output, label_batch=self.labDescVec, max_pool=maxpool)

        return logits
        # loss = None
        # if labels is not None:
        #     loss_fct = BCEWithLogitsLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        #
        #
        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
