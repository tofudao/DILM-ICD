from src.train_load import *
from src.roberta_model import *
from src.semble_model import *
from src.args_parse import *
from src.vocab_all import *
from src.process import *
from src.evaluation import all_metrics

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
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


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
    print(remove_columns)

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

    def getitem(examples):
        # Tokenize the texts
        label_list = []
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
        return result

    def data_collator_train(features):
        batch = dict()

        # ==========================================================================================================
        if args.dataEnhance:
            for i in range(len(features)):  # 数据增强
                len_fea = int(len(features[i]['input_ids'][1:-1]))
                if random.random() < args.dataEnhanceRatio / 2:  # 随机排列
                    features[i]['input_ids'][1:-1] = torch.tensor(features[i]['input_ids'])[1:-1][
                        np.random.permutation(len_fea)].tolist()
                if random.random() < args.dataEnhanceRatio:  # 逆置
                    features[i]['input_ids'][1:-1] = torch.tensor(features[i]['input_ids'])[1:-1][
                        range(len_fea)[::-1]].tolist()
        # ==========================================================================================================

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
        return batch

    def data_collator(features):
        batch = dict()
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
        return batch

    processed_datasets = raw_datasets.map(getitem, batched=True,
                                          remove_columns=remove_columns)  # fn_kwargs={"vocab": vocab}
    print(processed_datasets)

    train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=data_collator_train,
                                  batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    #     train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=data_collator_train,
    #                                   batch_size=args.batch_size)
    #     eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size)
    #     test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    train_dataloader = accelerator.prepare(train_dataloader)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epoch * num_update_steps_per_epoch

    # 一个周期的epoch数
    T_epochs = args.num_train_epochs

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    batch_id = 0

    print("这是最开始")
    filename = args.best_model_path + "label_vec.pkl"
    if not os.path.exists(filename):
        print("不存在文件")
        model = Roberta_model.from_pretrained(
            args.best_model_path,
            config=config,
            args=args,
            vocab=vocab
        )
        model = accelerator.prepare(model)
        model.eval()
        label_vec = dict()
        # ===
        all_labels = []
        att_list = []
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                outputs, att = model(**batch)
            all_labels.extend(batch["label_ids"])
            att_list.extend(att)
        label_vec['train_input_tensor'] = torch.stack(att_list)
        label_vec['train_target_tensor'] = torch.stack(all_labels)

        all_labels = []
        att_list = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs, att = model(**batch)
            all_labels.extend(batch["label_ids"])
            att_list.extend(att)
        label_vec['val_input_tensor'] = torch.stack(att_list)
        label_vec['val_target_tensor'] = torch.stack(all_labels)

        all_labels_batch = []
        att_list_batch = []
        all_preds_raw = []
        all_preds = []
        all_labels = []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs, att = model(**batch)
            all_labels_batch.extend(batch["label_ids"])
            att_list_batch.extend(att)
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
        label_vec['test_input_tensor'] = torch.stack(att_list_batch)
        label_vec['test_target_tensor'] = torch.stack(all_labels_batch)
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        print(
            f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(" ")
        with open(filename, 'wb') as file:
            # 从文件中加载序列化的对象
            pickle.dump(label_vec, file)
        del model
    else:
        print("存在文件")
        model = Roberta_model.from_pretrained(
            args.best_model_path,
            config=config,
            args=args,
            vocab=vocab
        )
        model = accelerator.prepare(model)
        model.eval()
        all_preds_raw=[]
        all_preds=[]
        all_labels=[]
        for step, batch in enumerate(test_dataloader):
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

        print(
            f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(" ")
        del model
    # =======================================================================================
    if args.use_finetune:
        check=torch.load(args.best_model_path+"check.pth",map_location=torch.device('cuda:1'))
        print()
        print("验证集最好的之前测试集最佳值",check['metrics'])
        # print(check['model_state_dict'])
        print()
        weight=check['model_state_dict']['attention.third_linears.weight']
        bias=check['model_state_dict']['attention.third_linears.bias']
    #==========================================================================================
    print("semble进行集成")
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data.unsqueeze(dim=1)
            self.labels = labels.unsqueeze(1)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    with open(filename, 'rb') as file:
        # 从文件中加载序列化的对象
        label_vec = pickle.load(file)
    input_tensor = label_vec['train_input_tensor']
    target_tensor = label_vec['train_target_tensor']
    val_input_tensor = label_vec['val_input_tensor']
    val_target_tensor = label_vec['val_target_tensor']
    test_input_tensor = label_vec['test_input_tensor']
    test_target_tensor = label_vec['test_target_tensor']
    # print(label_vec)
    print(label_vec['train_input_tensor'].shape)
    print(label_vec['train_target_tensor'].shape)
    print("sf")
    print(label_vec['val_input_tensor'].shape)
    print(label_vec['val_target_tensor'].shape)

    test_all_pred_raw = []
    test_all_true = []
    test_all_pred = []
    # 求第i个标签的weight
    # for i in range(len(target_tensor[1])):
    for i in range(4):
        print("第i轮", i)
        dataset = CustomDataset(input_tensor[:, i, :], target_tensor[:, i])

        # 创建 DataLoader 实例
        data_loader = DataLoader(dataset, batch_size=args.semble_batch, shuffle=True)
        # ====================================================
        dataset = CustomDataset(val_input_tensor[:, i, :], val_target_tensor[:, i])
        eval_dataloader = DataLoader(dataset, batch_size=args.semble_batch, shuffle=True)
        dataset = CustomDataset(test_input_tensor[:, i, :], test_target_tensor[:, i])
        test_dataloader = DataLoader(dataset, batch_size=args.semble_batch, shuffle=True)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        test_dataloader = accelerator.prepare(test_dataloader)

        # ==================================================================
        model_semble = Semble(size=args.d_a * 2, name='semble_{}'.format(i),weight=weight,bias=bias)

        # ========================
        # lr策略AdamW
        betas = (0.9, 0.999)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model_semble.parameters()),
                          lr=args.lr, betas=betas)
        # print("optimizer", optimizer)
        if args.use_lr_scheduler:
            itersPerEpoch = num_update_steps_per_epoch
            print("itersPerEpoch", itersPerEpoch)
            epoch = T_epochs
            warmupEpochs = args.warmup
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                           num_training_steps=epoch * itersPerEpoch)
        model_semble, optimizer, train_dataloader = accelerator.prepare(
            model_semble, optimizer, data_loader
        )
        criterions = nn.BCEWithLogitsLoss()
        for epoch in tqdm(range(args.add_epoch)):
            # print(" ")
            # print(datetime.datetime.now().strftime('%H:%M:%S'))
            # print("添加——第%d轮" % (epoch + 1))
            model_semble.train()
            optimizer.zero_grad()
            losses = []
            epoch_loss = 0.0
            #################################################################
            for step, batch in enumerate(train_dataloader):
                batch_id += 1
                # print("batch",batch)
                # #torch.Size([16, 768])
                # print(batch[0].shape)
                # #torch.Size([16])
                # print(batch[1].shape)
                output = model_semble(batch[0])
                # print(output.shape)
                loss = criterions(output, batch[1])
                loss = loss
                accelerator.backward(loss)
                losses.append(loss.item())
                epoch_loss += loss.item()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(loss=epoch_loss / batch_id)

            # model_semble.eval()
            # all_preds = []
            # all_preds_raw = []
            # all_labels = []
            # for step, batch in enumerate(eval_dataloader):

            #     with torch.no_grad():
            #         outputs = model_semble(batch[0])
            #     preds_raw = outputs.sigmoid().cpu()
            #     preds = (preds_raw > 0.5).int()
            #     all_preds_raw.extend(list(preds_raw))
            #     all_preds.extend(list(preds))
            #     all_labels.extend(list(batch[1].cpu().numpy()))
            # all_preds_raw = np.stack(all_preds_raw)
            # all_preds = np.stack(all_preds)
            # all_labels = np.stack(all_labels)
            # metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            # # print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
            # # print(
            # #     f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
            # # print(" ")

        # ==================================================================
        model_semble.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = model_semble(batch[0])
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch[1].cpu().numpy()))
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        # ============================================
        test_all_pred.append(all_preds.squeeze(1))
        test_all_pred_raw.append(all_preds_raw.squeeze(1))
        test_all_true.append(all_labels.squeeze(1))
        # ============================================
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
        print(
            f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(" ")
    all_preds = np.stack(test_all_pred).transpose()
    all_preds_raw = np.stack(test_all_pred_raw).transpose()
    all_labels = np.stack(test_all_true).transpose()

    print("all_preds.shape", all_preds.shape)
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
    print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
    print(
        f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    print(" ")


if __name__ == "__main__":
    main()

# train_load和rags_parse复制了 attion_layer 改了一点点pool
