
from src.train_load import *

from src.roberta_model import *
from src.get_label_model import *
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

import gc
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

    train_dataloader = DataLoader(processed_datasets["train"], collate_fn=data_collator,
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

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    max_train_steps = args.n_epoch * num_update_steps_per_epoch

    # 一个周期的epoch数
    T_epochs = args.num_train_epochs


    print("这是最开始")
    filename = args.best_model_path + "label_vec.pkl"

    #============================================================================================
    weight = None
    bias = None
    if args.use_finetune:
        check=torch.load(args.best_model_path+"check.pth", map_location=device)
        # check=torch.load(args.best_model_path+"check.pth",map_location=torch.device('cuda:2'))
        # check = torch.load(args.best_model_path + "check.pth")
        print()
        print("验证集最好的之前测试集最佳值",check['metrics'])
        # print(check['model_state_dict'])
        print()
        weight=check['model_state_dict']['attention.third_linears.weight']
        bias=check['model_state_dict']['attention.third_linears.bias']
        del check
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


    test_all_pred_raw_pre = []
    test_all_true_pre = []
    test_all_pred_pre = []
    test_all_pred_raw_after = []
    test_all_true_after = []
    test_all_pred_after = []
    # 求第i个标签的weight
    model = Roberta_model.from_pretrained(
        args.best_model_path,
        config=config,
        args=args,
        vocab=vocab
    )
    model = accelerator.prepare(model)

    gap=args.lable_gap
    iter_label=math.ceil(num_labels / gap)

    long_list=list(range(num_labels))
    # num_lists = num_labels // gap
    # 使用列表切片来分成多个间隔50的列表
    short_lists = [long_list[i:i + gap] for i in range(0, num_labels, gap)]

    # for i in range(num_labels):
    for j in range(iter_label):
    # for j in range(1):
        #====================================================
        # for i in range(4):
        print("第j轮", j)

        #=========================================
        model.eval()
        label_vec = dict()
        # ===
        all_labels = []
        att_list = []
        print("训练集开始",datetime.datetime.now().strftime('%H:%M:%S'))
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                outputs, att = model(**batch)
            all_labels.extend(batch["label_ids"][:,short_lists[j]].cpu())
            att_list.extend(att[:,short_lists[j],:].cpu())
        label_vec['train_input_tensor'] = torch.stack(att_list)
        label_vec['train_target_tensor'] = torch.stack(all_labels)
        print("label_vec[train_input_tensor]",label_vec['train_input_tensor'].shape)
        print("label_vec[train_target_tensor]",label_vec['train_target_tensor'].shape)

        print("验证集开始", datetime.datetime.now().strftime('%H:%M:%S'))
        all_labels = []
        att_list = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs, att = model(**batch)
            all_labels.extend(batch["label_ids"][:,short_lists[j]].cpu())
            att_list.extend(att[:,short_lists[j],:].cpu())
        label_vec['val_input_tensor'] = torch.stack(att_list)
        label_vec['val_target_tensor'] = torch.stack(all_labels)

        print("测试集开始",datetime.datetime.now().strftime('%H:%M:%S'))
        all_labels = []
        att_list = []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs, att = model(**batch)
            all_labels.extend(batch["label_ids"][:,short_lists[j]].cpu())
            # torch.Size([32, 50, 768])
            att_list.extend(att[:,short_lists[j],:].cpu())
        label_vec['test_input_tensor'] = torch.stack(att_list)
        label_vec['test_target_tensor'] = torch.stack(all_labels)
        print("测试集结束", datetime.datetime.now().strftime('%H:%M:%S'))
        # with open(filename, 'wb') as file:
        #     # 从文件中加载序列化的对象
        #     pickle.dump(label_vec, file)
        #===========================
        # outputs=None
        # att=None
        # all_labels=[]
        # att_list=[]
        #=======================
        # del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # #================================================
        # print(datetime.datetime.now().strftime('%H:%M:%S'))
        # with open(filename, 'rb') as file:
        #     # 从文件中加载序列化的对象
        #     label_vec = pickle.load(file)
        input_tensor = label_vec['train_input_tensor']
        target_tensor = label_vec['train_target_tensor']
        val_input_tensor = label_vec['val_input_tensor']
        val_target_tensor = label_vec['val_target_tensor']
        test_input_tensor = label_vec['test_input_tensor']
        test_target_tensor = label_vec['test_target_tensor']
        print(input_tensor.shape)
        print("存储结束",datetime.datetime.now().strftime('%H:%M:%S'))

        # 求第i个标签的weight
        list_short_x = list(range(len(short_lists[j])))
        for k in range(len(short_lists[j])):
        # for k in range(4):
            ind_x=list_short_x[k]
            i=short_lists[j][k]
            print("第i个标签",i)
            dataset = CustomDataset(input_tensor[:, ind_x, :], target_tensor[:, ind_x])
            # 创建 DataLoader 实例
            train_semble_dataloader = DataLoader(dataset, batch_size=args.semble_batch, shuffle=True)
            # ====================================================
            dataset = CustomDataset(val_input_tensor[:, ind_x, :], val_target_tensor[:, ind_x])
            eval_semble_dataloader = DataLoader(dataset, batch_size=args.semble_batch)
            dataset = CustomDataset(test_input_tensor[:, ind_x, :], test_target_tensor[:, ind_x])
            test_semble_dataloader = DataLoader(dataset, batch_size=args.semble_batch)

            print("清空")
            dataset=None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            eval_semble_dataloader = accelerator.prepare(eval_semble_dataloader)
            test_semble_dataloader = accelerator.prepare(test_semble_dataloader)

            # ==================================================================
            # # print("weight",weight)
            # # print("bias",bias)
            # # model_semble = Semble(size=args.d_a * 2, label_name=i,model_state=model_state,weight=weight[i,:],bias=bias[i])
            model_semble = Semble(size=args.d_a * 2, label_name=i, weight=weight[i,:],bias=bias[i])
            # ========================
            # lr策略AdamW
            betas = (0.9, 0.999)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model_semble.parameters()),
                              lr=args.lr, betas=betas)
            # print("optimizer", optimizer)
            if args.use_lr_scheduler:
                itersPerEpoch = len(train_semble_dataloader)
                print("itersPerEpoch", itersPerEpoch)
                epoch = T_epochs
                warmupEpochs = args.warmup
                lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                               num_training_steps=epoch * itersPerEpoch)

            model_semble, optimizer, train_semble_dataloader = accelerator.prepare(
                model_semble, optimizer, train_semble_dataloader
            )
            criterions = nn.BCEWithLogitsLoss()
            ######################################################
            #

            print("开始前")
            model_semble.eval()
            all_preds = []
            all_preds_raw = []
            all_labels = []
            for step, batch in enumerate(test_semble_dataloader):
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
            print("all_labels.shape",all_labels.shape)
            # ============================================
            print("all_preds",all_preds.shape)
            print("all_preds_raw",all_preds_raw.shape)
            print("all_labels",all_labels.shape)
            test_all_pred_pre.append(all_preds.squeeze(1))
            test_all_pred_raw_pre.append(all_preds_raw.squeeze(1))
            test_all_true_pre.append(all_labels.squeeze(1))
            # ============================================
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            # print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
            print(
                f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
            print(" ")
            ######################################################
            num_update_steps_per_epoch=math.ceil(len(train_semble_dataloader))
            max_train_steps = args.n_epoch * num_update_steps_per_epoch
            progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
            print(datetime.datetime.now().strftime('%H:%M:%S'))
            checkpoint={}
            batch_id = 0
            # =====================
            metrics_max = None
            metrics_max_val = -1
            metrics_max_auc = -1
            for epoch in tqdm(range(args.add_epoch)):
                # print(" ")
                # print(datetime.datetime.now().strftime('%H:%M:%S'))
                print("添加——第%d轮" % (epoch + 1))
                model_semble.train()
                optimizer.zero_grad()
                losses = []
                epoch_loss = 0.0
                #################################################################
                for step, batch in enumerate(train_semble_dataloader):
                    batch_id += 1
                    # print("batch",batch)
                    # #torch.Size([16, 768])
                    # print(batch[0].shape)
                    # #torch.Size([16])
                    # print(batch[1].shape)
                    output = model_semble(batch[0])
                    # print(output.shape)
                    loss = criterions(output, batch[1])

                    accelerator.backward(loss)
                    losses.append(loss.item())
                    epoch_loss += loss.item()

                    optimizer.step()
                    # lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=epoch_loss / batch_id)
                model_semble.eval()
                all_preds = []
                all_preds_raw = []
                all_labels = []
                for step, batch in enumerate(eval_semble_dataloader):
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

                metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
                metrics_val = metrics
                print(
                    f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
                print(" ")
                model_semble.eval()
                all_preds = []
                all_preds_raw = []
                all_labels = []
                for step, batch in enumerate(test_semble_dataloader):
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
                metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
                metrics_test=metrics
                # print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
                print(
                    f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
                print(" ")
                ##########################################
                print("当前验证集最大的值metrics_max_val",metrics_max_val)
                print("本轮得到的验证集的值metrics_val_f1",metrics_val['f1_micro'])
                # ================================================================
                #看auc
                if metrics_max is None:
                    metrics_max = metrics_test
                    metrics_max_val = metrics_val['f1_macro']
                    metrics_max_auc = metrics_val['auc_micro']
                    checkpoint = {'the epoch': epoch,
                                  'metrics': metrics_max,
                                  'model_state_dict': model_semble.state_dict()}
                    torch.save(checkpoint, args.best_model_path + 'sembel/check_{}.pth'.format(i))
                else:
                    if metrics_max_auc < metrics_val['auc_micro']:
                        print("进入,第i个进行存储", epoch + 1)
                        metrics_max_val = metrics_val['f1_macro']
                        metrics_max_auc = metrics_val['auc_micro']
                        metrics_max = metrics_test
                        checkpoint = {'the epoch': epoch,
                                      'metrics': metrics_max,
                                      'model_state_dict': model_semble.state_dict()}
                        torch.save(checkpoint, args.best_model_path + 'sembel/check_{}.pth'.format(i))
                # ================================================================
                # # ================================================================
                # #看f1
                # if metrics_max is None:
                #     metrics_max = metrics_test
                #     metrics_max_val = metrics_val['f1_macro']
                #     metrics_max_auc = metrics_val['auc_micro']
                #     checkpoint = {'the epoch': epoch,
                #                   'metrics': metrics_max,
                #                   'model_state_dict': model_semble.state_dict()}
                #     torch.save(checkpoint, args.best_model_path + 'sembel/check_{}.pth'.format(i))
                # else:
                #     if metrics_max_val < metrics_val['f1_macro']:
                #         print("进入,第i个进行存储", epoch + 1)
                #         metrics_max_val = metrics_val['f1_macro']
                #         metrics_max_auc = metrics_val['auc_micro']
                #         metrics_max = metrics_test
                #         checkpoint = {'the epoch': epoch,
                #                       'metrics': metrics_max,
                #                       'model_state_dict': model_semble.state_dict()}
                #         torch.save(checkpoint, args.best_model_path + 'sembel/check_{}.pth'.format(i))
                # # ================================================================
            #===========================
            # ==================================================================
            print("第i个搭到最佳",checkpoint['the epoch'])
            print(datetime.datetime.now().strftime('%H:%M:%S'))
            checkpoint = torch.load(args.best_model_path + 'sembel/check_{}.pth'.format(i), map_location=device)
            model_semble.load_state_dict(checkpoint['model_state_dict'])

            model_semble.eval()
            all_preds_raw = []
            all_labels = []
            all_preds = []

            for step, batch in enumerate(test_semble_dataloader):
                # print("数据参数", batch.device)
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
            # # ============================================
            test_all_pred_after.append(all_preds.squeeze(1))
            test_all_pred_raw_after.append(all_preds_raw.squeeze(1))
            test_all_true_after.append(all_labels.squeeze(1))
            # # ============================================
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            print(
                f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
            print(" ")
            lr_scheduler=None
            optimizer=None
            losses=[]
            model_semble=None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # =========================================================================
    test_all_pred_pre = np.stack(test_all_pred_pre).transpose()
    test_all_pred_raw_pre = np.stack(test_all_pred_raw_pre).transpose()
    test_all_true_pre = np.stack(test_all_true_pre).transpose()
    print("all_preds2.shape", test_all_pred_pre.shape)
    print("all_labels2.shape", test_all_pred_raw_pre.shape)
    print("all_preds_raw2.shape", test_all_true_pre.shape)
    metrics = all_metrics(yhat=test_all_pred_pre, y=test_all_true_pre, yhat_raw=test_all_pred_raw_pre)
    print(
        f"验证集开始时: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    print(" ")
    # ========================================================================
    test_all_pred_after = np.stack(test_all_pred_after).transpose()
    test_all_pred_raw_after = np.stack(test_all_pred_raw_after).transpose()
    test_all_true_after = np.stack(test_all_true_after).transpose()

    print("all_preds.shape", test_all_pred_after.shape)
    metrics = all_metrics(yhat=test_all_pred_after, y=test_all_true_after, yhat_raw=test_all_pred_raw_after)
    print(
        f"验证集结束后: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    print(" ")


if __name__ == "__main__":
    main()

# train_load和rags_parse复制了 attion_layer 改了一点点pool
