from src.train_load import *
# from src.roberta_model_moe import *
from src.moe_rnn_model import *
from src.args_parse import *
from src.vocab_all import *
from src.process import *
# from src.evaluation import all_metrics,find_theta
from src.evaluation import *

from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW, get_scheduler, AutoConfig
from transformers import AutoTokenizer, RobertaModel
from tqdm.autonotebook import tqdm
import random
import logging
import math
import torch
import datetime

logger = logging.getLogger(__name__)

# set the random seed if needed, disable by default
set_random_seed(random_seed=42)


def main():
    args = create_args_parser()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    raw_datasets = load_data(args.data_dir)
    print(raw_datasets)
    vocab = Vocab(args, raw_datasets)
    label_to_id = vocab.label_to_id
    num_labels = len(label_to_id)

    print(len(vocab.label_dict['train']))
    print(len(vocab.label_dict['valid']))
    print(len(vocab.label_dict['test']))
    remove_columns = raw_datasets["train"].column_names

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

    if "Text" in remove_columns:
        text_name = "Text"
    elif "text" in remove_columns:
        text_name = "text"
    if "Full_Labels" in remove_columns:
        label_columns = "Full_Labels"
    elif "target" in remove_columns:
        label_columns = "target"
    if 'Admission_Id' in remove_columns:
        id_admission = 'Admission_Id'
    elif '_id' in remove_columns:
        id_admission = '_id'

    def getitem(examples):
        # Tokenize the texts
        label_list = []
        id_list = [id for id in examples[id_admission]]

        texts = ((examples[text_name],))

        result = tokenizer(*texts, padding=False, max_length=args.max_seq_length, truncation=True,
                           add_special_tokens=True)
        batch_encoding = {"input_ids": result["input_ids"]}
        #         print("整个长度",len(result["input_ids"]))
        # batch_decoding = tokenizer.batch_decode(batch_encoding["input_ids"][1])
        if "Full_Labels" == label_columns:
            for labels in examples["Full_Labels"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split('|')])
        elif "target" == label_columns:
            for labels in examples["target"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split(',')])

        result["label_ids"] = label_list
        result['id_list'] = id_list
        return result

    def data_collator(features):
        batch = dict()
        ##############
        id_list = [feature['id_list'] for feature in features]

        max_length = max([len(f["input_ids"]) for f in features])
        # 最大长度 128
        #         print("最大长度",max_length)

        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
        label_ids = torch.zeros((len(features), num_labels))
        #         print("标签ID",label_ids.size())#[8, 8929]

        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["label_ids"] = label_ids
        #########
        batch['id_list'] = id_list
        return batch

    processed_datasets = raw_datasets.map(getitem, batched=True,
                                          remove_columns=remove_columns)  # fn_kwargs={"vocab": vocab}
    print(processed_datasets)

    # train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=data_collator_train,
    #                           batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    # train_dataloader = DataLoader(processed_datasets["train"], collate_fn=data_collator_train,
    #                               batch_size=args.batch_size)
    #     eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size)
    #     test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    ############################################################
    check = torch.load(args.best_model_path + "check.pth", map_location=torch.device('cuda:1'))
    print()
    print("验证集最好的之前测试集最佳值", check['metrics'])
    print("第i个epoch", check['epoch'])
    print()

    model = Roberta_model.from_pretrained(
        args.best_model_path,
        config=config,
        args=args,
        vocab=vocab
    )
    model = accelerator.prepare(model)
    # # 加载完整的模型状态
    # checkpoint = torch.load(args.best_model_path)
    # 顺便看测试集
    #####################################################

    metrics_max = None
    metrics_max_val = -1
    epoch_max = 0
    epoch_max_test = 0

    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
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
    print(" ")
    metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)

    # print("验证集")
    # # macro_map,the_first_f1,the_second_f1,micro_f1=find_theta(y=all_labels, yhat_raw=all_preds_raw)

    # f1_map=find_theta(y=all_labels, yhat_raw=all_preds_raw)

    # print(f1_map)
    # print("发发发")
    # f1_map=find_theta_2(y=all_labels, yhat_raw=all_preds_raw)

    # print(f1_map)
    # #============================================================
    # model.eval()
    # all_preds = []
    # all_preds_raw = []
    # all_labels = []
    # metrics_max = None
    # for step, batch in enumerate(test_dataloader):
    #     with torch.no_grad():
    #         if model.name == "plm_model":
    #             outputs = model(**batch)
    #         elif model.name == "rnn_model":
    #             outputs = model(batch)
    #     preds_raw = outputs.sigmoid().cpu()
    #     all_preds_raw.extend(list(preds_raw))
    #     all_labels.extend(list(batch["label_ids"].cpu().numpy()))

    # all_preds_raw = np.stack(all_preds_raw)
    # all_labels = np.stack(all_labels)
    # print("测试集")
    # macro_map,the_first_f1,the_second_f1,micro_f1=find_theta(y=all_labels, yhat_raw=all_preds_raw)
    # f1_map=find_theta(y=all_labels, yhat_raw=all_preds_raw)

    # print(f1_map)
    # print("发发发")
    # f1_map=find_theta_2(y=all_labels, yhat_raw=all_preds_raw)

    # print(f1_map)
    # #============================================================

    # #=========================
    # #测试集画图之后取最大值
    # micro_f1_list,macro_f1_list=get_test_f1(test_dataloader, model)
    # plot_f1(micro_f1_list,name='micro F1')
    # plot_f1(macro_f1_list,name='macro F1')
    # print("micro_f1_list",micro_f1_list)
    # print("macro_f1_list",macro_f1_list)
    # #=========================
    # # metrics_max_2,best_thre_2 = ans_test(test_dataloader, model)

    # metrics_max_2,best_thre_2 = ans_test_all(test_dataloader, model)
    # ##########################################
    # if metrics_max is None:
    #     metrics_max = metrics_max_2
    #     best_thre = best_thre_2
    # else:
    #     if metrics_max['f1_micro'] < metrics_max_2['f1_micro']:
    #         metrics_max = metrics_max_2
    #         best_thre = best_thre_2

    # print()
    # print("测试最好epoch:",check['epoch'])
    # print(f"epoch: {check['epoch']}, 最好的threshould:{best_thre:.2f}, ",",".join(f"{k}: {v:.4f}" for k, v in metrics_max.items()))
    # print()


if __name__ == "__main__":
    main()

# train_load和rags_parse复制了 attion_layer 改了一点点pool
