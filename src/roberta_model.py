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

class Roberta_model(RobertaPreTrainedModel):
    def __init__(self, config,args,vocab):
        super().__init__(config)
        self.name="plm_model"
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print(self.dropout)#Dropout(p=0.1, inplace=False)
        self.init_weights()

        self.plm_maxpool = nn.AdaptiveMaxPool2d((1, config.hidden_size))
        self.attention = AttentionLayer(args=args, vocab=vocab)
        ###############################
        if "text_label"==args.attention_mode:
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
        ########################################################################

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()
        #batch_size, num_chunks, chunk_size  torch.Size([2, 5, 2])
        print("batch_size, num_chunks, chunk_size ",input_ids.size())
        #input_ids.view(-1, chunk_size) torch.Size([10, 2])
        print("input_ids.view(-1, chunk_size)",input_ids.view(-1, chunk_size).size())
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
        #BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor([[[grad_fn=<NativeLayerNormBackward0>),
        # pooler_output=None, hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
        print("结果1", outputs)
        print("结果2",outputs[0])
        print("结果3",outputs[0].size())#torch.Size([5, 2, 768])
        ###############################
        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        maxpool = self.plm_maxpool(outputs[0].view(batch_size, num_chunks, -1)).detach_()


        logits=self.attention(x=hidden_output,label_batch=self.labDescVec,max_pool=maxpool)

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
