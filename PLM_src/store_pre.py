from src.train_load import *
from src.roberta_model import *
# from src.roberta_model import *
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
import pickle

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

    train_dataloader = DataLoader(processed_datasets["train"], collate_fn=data_collator,
                                  batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    # train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=data_collator_train,
    #                               batch_size=args.batch_size)
    #     eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size)
    #     test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    train_dataloader = accelerator.prepare(train_dataloader)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    ############################################################
    check = torch.load(args.best_model_path + "check.pth", map_location=torch.device('cuda:0'))
    print()
    print("验证集最好的之前测试集最佳值", check['metrics'])
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

    data_all = dict()
    #####################################################

    model.eval()
    all_preds_raw = []
    all_labels = []
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        preds_raw = outputs.sigmoid().cpu()
        all_preds_raw.extend(list(preds_raw))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_labels = np.stack(all_labels)
    # metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
    data_all['train_pred_label'] = all_preds_raw
    data_all['train_true_lable'] = all_labels
    # ============================================================

    #####################################################
    model.eval()
    all_preds_raw = []
    all_labels = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        preds_raw = outputs.sigmoid().cpu()
        all_preds_raw.extend(list(preds_raw))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_labels = np.stack(all_labels)
    data_all['val_pred_label'] = all_preds_raw
    data_all['val_true_lable'] = all_labels
    # ============================================================
    model.eval()
    all_preds_raw = []
    all_labels = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if model.name == "plm_model":
                outputs = model(**batch)
            elif model.name == "rnn_model":
                outputs = model(batch)
        preds_raw = outputs.sigmoid().cpu()
        all_preds_raw.extend(list(preds_raw))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))

    all_preds_raw = np.stack(all_preds_raw)
    all_labels = np.stack(all_labels)
    data_all['test_pred_label'] = all_preds_raw
    data_all['test_true_lable'] = all_labels
    # ============================================================
    with open('label_all.pkl', 'wb') as f:
        pickle.dump(data_all, f)

if __name__ == "__main__":
    main()

# train_load和rags_parse复制了 attion_layer 改了一点点pool
