from src.train_load import *
from src.get_new_data import *
from src.vocab_all import *
from src.args_parse import *
from src.process import *

from src.model import *

from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW,get_scheduler
import random
import math
import datetime


def main():
    args = create_args_parser()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # print(args)
    #取feather中的数据
    # data_pa, code_system = get_data_clean(direct=None, data_filename=None, split_filename=None, code_column_names=None)
    raw_datasets = load_data(args.data_dir)
    vocab = Vocab(args,raw_datasets)
    label_to_id=vocab.label_to_id
    num_labels = len(label_to_id)
#     print(label_to_id)
    print(len(vocab.label_dict['train']))
    print(len(vocab.label_dict['valid']))
    print(len(vocab.label_dict['test']))

    remove_columns = raw_datasets["train"].column_names
    print(remove_columns)

    if "Text" in remove_columns:
        text_name="Text"
    elif "text" in remove_columns:
        text_name = "text"
    if "Full_Labels" in remove_columns:
        label_columns="Full_Labels"
    elif "target" in remove_columns:
        label_columns = "target"
    if 'Admission_Id' in remove_columns:
        id_admission='Admission_Id'
    elif '_id' in remove_columns:
        id_admission = '_id'

    def getitem(examples):

        result = dict()
        input_list = []
        label_list = []
        id_list=[id for id in examples['_id']]

        for text in examples[text_name]:

            text_list = text.split()
            if len(text_list) > args.max_seq_length:
                text_list = text_list[:args.max_seq_length]
            input_list.append([vocab.word2index[word] for word in text_list])

        if "Full_Labels"==label_columns:
            for labels in examples["Full_Labels"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split('|')])
        elif "target"==label_columns:
            for labels in examples["target"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split(',')])

        result["input_ids"] = input_list
        result["label_ids"] = label_list
        result['id_list']=id_list

        return result

    def data_collator(features):
        #############
        id_list = [feature['id_list'] for feature in features]

        lenth_list = [len(feature['input_ids']) for feature in features]
        length_list = []
        input_list = []
        label_list = []
        batch = dict()
        sorted_indices = sorted(range(len(features)), key=lambda i: len(features[i]['input_ids']), reverse=True)

        for i in sorted_indices:  # 数据增强
            len_fea = int(len(features[i]['input_ids']))
            length_list.append(len_fea)

            input_list.append(torch.tensor(features[i]['input_ids']))
            label_tensor = torch.zeros(vocab.label_num)
            for j in features[i]['label_ids']:
                label_tensor[j] = 1
            label_list.append(label_tensor)
        # pad一下
        padded_batch = pad_sequence(input_list, batch_first=True)
        batch['input_ids'] = torch.LongTensor(padded_batch)
        batch['label_ids'] = torch.tensor([label_bat.tolist() for label_bat in label_list])
        batch['length_input'] = torch.tensor(length_list)
        #############
        batch['id_list'] = torch.tensor(id_list)
        #         batch["labDescVec"] = labDescVec
        return batch

    processed_datasets = raw_datasets.map(getitem, batched=True,
                                          remove_columns=remove_columns)  # fn_kwargs={"vocab": vocab}
    print(processed_datasets)

#     train_dataloader = DataLoader(processed_datasets["train"], shuffle=True,collate_fn=data_collator_train, batch_size=args.batch_size, pin_memory=True)
#     eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
#     test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
#     train_dataloader = DataLoader(processed_datasets["train"], shuffle=True,collate_fn=data_collator_train, batch_size=args.batch_size)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    model = RNN(args, vocab)
    if args.optimiser.lower() == "adamw":
        betas = (0.9, 0.999)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, betas=betas, weight_decay=args.weight_decay)
        print("optimizer", optimizer)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer
    )

    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)


    metrics_max = None
    metrics_max_val=-1
    epoch_max=0
    epoch_max_test=0
    ############################################################
    best_model = torch.load(args.best_model_path+"check.pth")

    # model.to(device)
    model.load_state_dict(best_model['model_state_dict'])

    # # 加载完整的模型状态
    # checkpoint = torch.load(args.best_model_path)
    #顺便看测试集
    #####################################################

    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    for step, batch in enumerate(eval_dataloader):
#             batch['length_input'] = batch['length_input'].to('cpu')
        with torch.no_grad():
            outputs = model(batch)
        preds_raw = outputs.sigmoid().cpu()
        preds = (preds_raw > 0.5).int()
        all_preds_raw.extend(list(preds_raw))
        all_preds.extend(list(preds))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)

    # print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
    print(
        f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    #         print("验证集: F1_micro: ", metrics['f1_micro'], "F1_macro: ", metrics['f1_macro'], "auc_micro: ", metrics['auc_micro'])
    #         print("prec_at_8: ", metrics['prec_at_8'],"auc_macro: ", metrics['auc_macro'])
    print(" ")

    metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)
    metrics_max_2,best_thre_2 = ans_test_all(test_dataloader, model)

    if metrics_max is None:
        metrics_max = metrics_max_2
        best_thre = best_thre_2
    else:
        if metrics_max['f1_micro'] < metrics_max_2['f1_micro']:
            metrics_max = metrics_max_2
            best_thre = best_thre_2
    print()
    print("验证集最好epoch:",epoch_max)
    print(f"epoch: {epoch_max_test}, 最好的threshould:{best_thre:.2f}, ",",".join(f"{k}: {v:.4f}" for k, v in metrics_max.items()))
    print()



if __name__ == "__main__":
    main()