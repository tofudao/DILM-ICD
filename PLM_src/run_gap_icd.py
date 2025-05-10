from src.train_load import *
from src.roberta_model_moe import *
# from src.roberta_model import *
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
    print(remove_columns)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    if "roberta" == args.model_type:
        model = Roberta_model.from_pretrained(args.model_name_or_path, config=config, args=args, vocab=vocab)
    #     print(model)

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

    # connect_token=int(args.max_seq_length/args.chunk_size/chunk_num)
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
        # max_length=args.max_seq_length
        # 最大长度 128
        #         print("最大长度",max_length)

        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        list_input_ids = [
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]


        original_input_order = list(range(max_length))
        if args.gap_type==0:
            batch["input_ids"] = torch.tensor(list_input_ids).contiguous().view((len(features), -1, args.chunk_size))
            batch["inverse_input_list"]=original_input_order
        # #==============================================================、
        # #增加一种 方法3
        elif args.gap_type==1:
            chunk_num=max_length // args.chunk_size
            gap_list = []
            for sublist in list_input_ids:
                sublist_chunks = [sublist[i::chunk_num] for i in range(chunk_num)]
                gap_list.append(sublist_chunks)
            batch["input_ids"]=torch.tensor(gap_list)
            # =====================================================================================
            original_input_list = [original_input_order[i::chunk_num] for i in range(chunk_num)]
            original_input_list=sum(original_input_list, [])
            inverse_indices = {index: i for i, index in enumerate(original_input_list)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            # original_sorted_idx=[sorted_idx for sorted_idx, original_idx in enumerate(original_input_list)]
            batch["inverse_input_list"]=inverse_indices_list
            # =====================================================================================
        #===============================================================
        # #==============================================================、
        # #增加一种 方法2 合并相连4个再gap
        elif args.gap_type == 2:
            chunk_num=max_length // args.chunk_size
            # connect_token=args.chunk_size//chunk_num
            connect_token=args.connect_token
            connect_gap=args.chunk_size//connect_token
            # gap_list = [list_input_ids[i::chunk_num] for i in range(chunk_num)]
            gap_list = []
            # 遍历 long_list 中的每个子列表，并将它们分割成128个元素的块
            for sublist in list_input_ids:
                #切割connect_token个连续token
                sublist_chunks = [sublist[i:i+connect_token] for i in range(0,max_length,connect_token)]
                # 使用列表推导式和步长为32的切片来创建块
                sublist_chunks = [sublist_chunks[i::chunk_num] for i in range(chunk_num)]
                sublist_chunks = torch.tensor(sublist_chunks).reshape(chunk_num, args.chunk_size)
                gap_list.append(sublist_chunks.tolist())
            batch["input_ids"] = torch.tensor(gap_list)
            # ====================================================================================
            connect_input_list=[original_input_order[i: i + connect_token] for i in range(0, max_length, connect_token)]
            original_input_list = [connect_input_list[i::chunk_num] for i in range(chunk_num)]
            # input_index_chunks = list(torch.tensor(original_input_list).reshape(chunk_num, args.chunk_size))
            input_index=sum(sum(original_input_list, []), [])
            inverse_indices = {index: i for i, index in enumerate(input_index)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            # original_sorted_idx=[sorted_idx for sorted_idx, original_idx in enumerate(original_input_list)]
            batch["inverse_input_list"]=inverse_indices_list
            # ====================================================================================
            # batch["gap_list"]=torch.tensor(gap_list)
        # ===============================================================
        # #==============================================================、
        # #增加一种 方法3

        # chunk_num=max_length // args.chunk_size
        # gap_list = []
        # for sublist in list_input_ids:
        #     sublist_chunks = [sublist[i::args.chunk_size] for i in range(args.chunk_size)]
        #     gap_list.append(sublist_chunks)
        # batch["gap_list"]=torch.tensor(gap_list)
        # #===============================================================
        # #==============================================================、
        # #增加一种 方法2

        # chunk_num=max_length // args.chunk_size
        # # connect_token=args.chunk_size//chunk_num
        # connect_token=4
        # connect_gap=args.chunk_size//connect_token

        # # gap_list = [list_input_ids[i::chunk_num] for i in range(chunk_num)]
        # gap_list = []
        # # 遍历 long_list 中的每个子列表，并将它们分割成128个元素的块
        # for sublist in list_input_ids:
        #     # 使用列表推导式和步长为32的切片来创建块
        #     sublist_chunks = [sublist[i:i+connect_token] for i in range(0,max_length,connect_token)]
        #     # print(len(sublist_chunks))
        #     # for n,x in enumerate(sublist_chunks):
        #     #     print("nnn",n)
        #     #     print(torch.tensor(x).shape)
        #     # print(connect_gap) 32
        #     # sublist_chunks_list=[]
        #     # for i in range(chunk_num):
        #     #     print(len(sublist_chunks[i::chunk_num])
        #     #     sublist_chunks_list.append(sublist_chunks[i::chunk_num])
        #     sublist_chunks = [sublist_chunks[i::chunk_num] for i in range(chunk_num)]
        #     # for n,x in enumerate(sublist_chunks):
        #     #     print("nnn",n)
        #     #     print(torch.tensor(x).shape)
        #     # print(sublist_chunks)
        #     # print("sublist_chunks",torch.tensor(sublist_chunks).shape)
        #     sublist_chunks = torch.tensor(sublist_chunks).reshape(chunk_num, args.chunk_size)
        #     gap_list.append(sublist_chunks.tolist())
        # batch["gap_list"]=torch.tensor(gap_list)
        # # print(batch["gap_list"].shape)
        # #===============================================================
        # #==============================================================、
        # #增加一种 方法1

        # chunk_num=int(max_length / args.chunk_size)
        # # print(chunk_num)

        # # gap_list = [list_input_ids[i::chunk_num] for i in range(chunk_num)]
        # gap_list = []
        # # 遍历 long_list 中的每个子列表，并将它们分割成128个元素的块
        # for sublist in list_input_ids:
        #     # 使用列表推导式和步长为32的切片来创建块
        #     sublist_chunks = [sublist[i::chunk_num] for i in range(chunk_num)]
        #     gap_list.append(sublist_chunks)
        # batch["gap_list"]=torch.tensor(gap_list)
        # # print("batch[gap_list",batch["gap_list"].shape)
        # #===============================================================

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
        # max_length=args.max_seq_length
        # 最大长度 128
        #         print("最大长度",max_length)

        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        list_input_ids = [
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]

        original_input_order = list(range(max_length))
        if args.gap_type==0:
            batch["input_ids"] = torch.tensor(list_input_ids).contiguous().view((len(features), -1, args.chunk_size))
            batch["inverse_input_list"]=original_input_order
        # #==============================================================、
        # #增加一种 方法3
        elif args.gap_type==1:
            chunk_num=max_length // args.chunk_size
            gap_list = []
            for sublist in list_input_ids:
                sublist_chunks = [sublist[i::chunk_num] for i in range(chunk_num)]
                gap_list.append(sublist_chunks)
            batch["input_ids"]=torch.tensor(gap_list)
            # =====================================================================================
            original_input_list = [original_input_order[i::chunk_num] for i in range(chunk_num)]
            original_input_list=sum(original_input_list, [])
            inverse_indices = {index: i for i, index in enumerate(original_input_list)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            # original_sorted_idx=[sorted_idx for sorted_idx, original_idx in enumerate(original_input_list)]
            batch["inverse_input_list"]=inverse_indices_list
            # =====================================================================================
        #===============================================================
        # #==============================================================、
        # #增加一种 方法2 合并相连4个再gap
        elif args.gap_type == 2:
            chunk_num=max_length // args.chunk_size
            # connect_token=args.chunk_size//chunk_num
            connect_token=args.connect_token
            connect_gap=args.chunk_size//connect_token
            # gap_list = [list_input_ids[i::chunk_num] for i in range(chunk_num)]
            gap_list = []
            # 遍历 long_list 中的每个子列表，并将它们分割成128个元素的块
            for sublist in list_input_ids:
                #切割connect_token个连续token
                sublist_chunks = [sublist[i:i+connect_token] for i in range(0,max_length,connect_token)]
                # 使用列表推导式和步长为32的切片来创建块
                sublist_chunks = [sublist_chunks[i::chunk_num] for i in range(chunk_num)]
                sublist_chunks = torch.tensor(sublist_chunks).reshape(chunk_num, args.chunk_size)
                gap_list.append(sublist_chunks.tolist())
            batch["input_ids"] = torch.tensor(gap_list)
            # ====================================================================================
            connect_input_list=[original_input_order[i: i + connect_token] for i in range(0, max_length, connect_token)]
            original_input_list = [connect_input_list[i::chunk_num] for i in range(chunk_num)]
            # input_index_chunks = list(torch.tensor(original_input_list).reshape(chunk_num, args.chunk_size))
            input_index=sum(sum(original_input_list, []), [])
            inverse_indices = {index: i for i, index in enumerate(input_index)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            # original_sorted_idx=[sorted_idx for sorted_idx, original_idx in enumerate(original_input_list)]
            batch["inverse_input_list"]=inverse_indices_list
            # ====================================================================================
        # ==============================================================、
        # #增加一种

        # chunk_num=int(max_length / args.chunk_size)
        # # print(chunk_num)

        # # gap_list = [list_input_ids[i::chunk_num] for i in range(chunk_num)]
        # gap_list = []
        # # 遍历 long_list 中的每个子列表，并将它们分割成128个元素的块
        # for sublist in list_input_ids:
        #     # 使用列表推导式和步长为32的切片来创建块
        #     sublist_chunks = [sublist[i::chunk_num] for i in range(chunk_num)]
        #     gap_list.append(sublist_chunks)
        # batch["gap_list"]=torch.tensor(gap_list)
        # # print(batch["gap_list"].shape)
        # #===============================================================
        # # ==============================================================
        # # 增加一种 方法4 滑动
        # chunk_num = max_length // args.chunk_size
        # step = args.chunk_size // 2  # 滑动步长
        # gap_list = []
        # for sublist in list_input_ids:
        #     sublist_chunks = [sublist[i:i + args.chunk_size] for i in range(0, max_length - step, step)]
        #     # for n,x in enumerate(sublist_chunks):
        #     #     print("nnn",n)
        #     #     print(torch.tensor(x).shape)
        #     gap_list.append(sublist_chunks)
        # # print("结束")
        # batch["gap_list"] = torch.tensor(gap_list)
        # # ===============================================================
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

    if args.optimiser.lower() == "adamw":
        if args.use_different_lr:
            ignored_params = list(map(id, model.roberta.parameters()))  # 返回的是parameters的 内存地址
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer_grouped_parameters = [
                {
                    "params": model.roberta.parameters(),
                    # "lr":2e-5,
                    "lr": args.plm_lr,
                },
                {
                    "params": base_params,
                    "lr": args.lr,
                    # "lr": 0.0005,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        else:
            betas = (0.9, 0.999)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, betas=betas, weight_decay=args.weight_decay)
            print("optimizer", optimizer)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # 一个周期的epoch数
    T_epochs = args.num_train_epochs

    if args.use_lr_scheduler:
        itersPerEpoch = num_update_steps_per_epoch
        print("itersPerEpoch", itersPerEpoch)
        epoch = T_epochs
        warmupEpochs = args.warmup
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                       num_training_steps=epoch * itersPerEpoch)
    #     lr_scheduler = get_scheduler(
    #             name='linear',
    #             optimizer=optimizer,
    #             num_warmup_steps=2000,
    #             num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    #         )
    criterions = nn.BCEWithLogitsLoss()
    print("optimizer", optimizer)
    print("lr_scheduler", lr_scheduler)

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    batch_id = 0
    metrics_max = None
    metrics_max_val = -1
    epoch_max = 0
    epoch_max_test = 0

    for epoch in tqdm(range(args.n_epoch)):
        print(" ")
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        print("第%d轮" % (epoch + 1))
        model.train()
        optimizer.zero_grad()
        losses = []
        epoch_loss = 0.0
        #################################################################
        for step, batch in enumerate(train_dataloader):
            batch_id += 1
            outputs,att = model(**batch)
            loss = criterions(outputs, batch['label_ids'])
            # loss =criterions(output.view(-1, vocab.label_num),label_batch.view(-1, vocab.label_num))
            # print("第三方",loss.grad_fn)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            losses.append(loss.item())
            epoch_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(loss=epoch_loss / batch_id)
        #################################################
        # for step, batch in enumerate(train_dataloader):
        #     batch_id += 1
        #     with autocast():
        #         output = model(**batch)
        #         loss = criterions(output, batch['labels'])
        #         # loss =criterions(output.view(-1, vocab.label_num),label_batch.view(-1, vocab.label_num))
        #     loss = loss / args.gradient_accumulation_steps
        #     scaler.scale(loss).backward()
        #     losses.append(loss.item())
        #     epoch_loss += loss.item()
        #     if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
        #         scaler.step(optimizer)
        #         scaler.update()
        #         optimizer.zero_grad()
        #         lr_scheduler.step()
        #         progress_bar.update(1)
        #         progress_bar.set_postfix(loss=epoch_loss / batch_id)
        #################################################

        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs,att = model(**batch)
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]),
              "学习率lr{}:".format(optimizer.param_groups[1]["lr"]), "loss: ", np.mean(losses).item())
        print(
            f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(" ")
        #         ############################################################
        #         #存储
        #         if metrics_max_val<metrics['f1_micro']:
        #             epoch_max=epoch+1
        #             metrics_max_val = metrics['f1_micro']
        #             if args.best_model_path is not None:
        #                 if args.best_model_path is not None:
        #                     os.makedirs(args.best_model_path, exist_ok=True)
        #                 accelerator.wait_for_everyone()
        #                 unwrapped_model = accelerator.unwrap_model(model)
        #                 unwrapped_model.save_pretrained(args.best_model_path, save_function=accelerator.save)
        #         #顺便看测试集
        #         #####################################################
        metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)
        #         metrics_max_2,best_thre_2 = ans_test_all(test_dataloader, model)
        ##########################################
        if metrics_max is None:
            metrics_max = metrics_max_2
            best_thre = best_thre_2
        else:
            if metrics_max['f1_micro'] < metrics_max_2['f1_micro']:
                epoch_max_test = epoch + 1
                metrics_max = metrics_max_2
                best_thre = best_thre_2
        ############################################################
        # 存储
        if metrics_max_val < metrics['f1_micro']:
            epoch_max = epoch + 1
            metrics_max_val = metrics['f1_micro']

            if args.best_model_path is not None:
                os.makedirs(args.best_model_path, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.best_model_path, save_function=accelerator.save)
            ###
            checkpoint = {'epoch': epoch + 1,
                          'metrics': metrics_max,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint, args.best_model_path + "check.pth")
        # 顺便看测试集
        #####################################################
    print()
    print("验证集最好epoch:", epoch_max)
    print(f"epoch: {epoch_max_test}, 最好的threshould:{best_thre:.2f}, ",
          ",".join(f"{k}: {v:.4f}" for k, v in metrics_max.items()))
    print()


if __name__ == "__main__":
    main()

# train_load和rags_parse复制了 attion_layer 改了一点点pool
