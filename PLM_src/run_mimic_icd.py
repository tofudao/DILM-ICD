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

    def getitem(examples):

        result = dict()
        input_list = []
        label_list = []



        for text in examples[text_name]:
            text_list = text.split()
            if len(text_list) > args.max_seq_length:
                text_list = text_list[:args.max_seq_length]
#             elif len(text_list) < args.max_seq_length:
#                 for i in range(args.max_seq_length-len(text_list)):
#                     text_list.append('_PAD')
#             print(len(text_list))
            input_list.append([vocab.word2index[word] for word in text_list])

        if "Full_Labels"==label_columns:
            for labels in examples["Full_Labels"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split('|')])
        elif "target"==label_columns:
            for labels in examples["target"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split(',')])

        result["input_ids"] = input_list
        result["label_ids"] = label_list

        return result

    def data_collator_train(features):
        lenth_list = [len(feature['input_ids']) for feature in features]
        length_list = []
        input_list = []
        label_list = []
        batch = dict()
        sorted_indices = sorted(range(len(features)), key=lambda i: len(features[i]['input_ids']), reverse=True)

        for i in sorted_indices:  # 数据增强
            len_fea = int(len(features[i]['input_ids']))
            length_list.append(len_fea)
            # ==========================================================================================================
            if args.dataEnhance:
                if random.random() < args.dataEnhanceRatio / 2:  # 随机排列
                    features[i]['input_ids'][:len_fea] = torch.tensor(features[i]['input_ids'][:len_fea])[
                        np.random.permutation(len_fea)].tolist()
                if random.random() < args.dataEnhanceRatio:  # 逆置
                    features[i]['input_ids'][:len_fea] = torch.tensor(features[i]['input_ids'][:len_fea])[
                        range(len_fea)[::-1]].tolist()
            # ==========================================================================================================
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
        #         batch["labDescVec"] = labDescVec
        return batch

    def data_collator(features):
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
        #         batch["labDescVec"] = labDescVec
        return batch

    processed_datasets = raw_datasets.map(getitem, batched=True,
                                          remove_columns=remove_columns)  # fn_kwargs={"vocab": vocab}
    print(processed_datasets)

#     train_dataloader = DataLoader(processed_datasets["train"], shuffle=True,collate_fn=data_collator_train, batch_size=args.batch_size, pin_memory=True)
#     eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
#     test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size, pin_memory=True)
    train_dataloader = DataLoader(processed_datasets["train"], shuffle=True,collate_fn=data_collator_train, batch_size=args.batch_size)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    model = RNN(args, vocab)
    if args.optimiser.lower() == "adamw":
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

    if args.use_lr_scheduler:
        # ================================================
        # itersPerEpoch = len(train_dataloader)
        itersPerEpoch = num_update_steps_per_epoch
        print("itersPerEpoch", itersPerEpoch)
        epoch = args.num_train_epochs
        #         epoch = 18
        warmupEpochs = args.warmup
        #         warmupEpochs = 2
        #         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,6, 2)
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
    #     print("学习率{}:".format(optimizer.param_groups[0]))
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    batch_id = 0
    metrics_max = None
    metrics_max_val=-1
    epoch_max=0
    epoch_max_test=0

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
#             batch['length_input'] = batch['length_input'].to('cpu')
            output = model(batch)
            loss = criterions(output, batch['label_ids'])
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
        #     batch['length_input'] = batch['length_input'].to('cpu')
        #     with autocast():
        #         output = model(batch)
        #         loss = criterions(output, batch['label_ids'])
        #         # loss =criterions(output.view(-1, vocab.label_num),label_batch.view(-1, vocab.label_num))
        #     scaler.scale(loss).backward()
        #     losses.append(loss.item())
        #     epoch_loss += loss.item()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()
        #     lr_scheduler.step()
        #     progress_bar.update(1)
        #     progress_bar.set_postfix(loss=epoch_loss / batch_id)
        #     # lr_scheduler.step(index+batch_id/self.train_len)
        # #################################################################
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

        print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]), "loss: ", np.mean(losses).item())
        print(
            f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        #         print("验证集: F1_micro: ", metrics['f1_micro'], "F1_macro: ", metrics['f1_macro'], "auc_micro: ", metrics['auc_micro'])
        #         print("prec_at_8: ", metrics['prec_at_8'],"auc_macro: ", metrics['auc_macro'])
        print(" ")
        # ############################################################
        # # 存储
        # if metrics_max_val < metrics['f1_micro']:
        #     epoch_max = epoch + 1
        #     metrics_max_val = metrics['f1_micro']
        #     if args.best_model_path is not None:
        #         os.makedirs(args.best_model_path, exist_ok=True)
        #         # 包括模型参数及额外状态信息
        #         checkpoint = {'epoch': epoch + 1,
        #                       'metrics': metrics_max,
        #                       'model_state_dict': model.state_dict(),
        #                       'optimizer_state_dict': optimizer.state_dict()}
        #         torch.save(checkpoint, args.best_model_path + "check.pth")
        #     # # 加载完整的模型状态
        #     # checkpoint = torch.load(args.best_model_path)
        # # 顺便看测试集
        # #####################################################
        #
        metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)
#         metrics_max_2,best_thre_2 = ans_test_all(test_dataloader, model)
        ##########################################
        if metrics_max is None:
            metrics_max = metrics_max_2
            best_thre = best_thre_2
        else:
            if metrics_max['f1_micro'] < metrics_max_2['f1_micro']:
                epoch_max_test=epoch+1
                metrics_max = metrics_max_2
                best_thre = best_thre_2
        ############################################################
        # 存储
        if metrics_max_val < metrics['f1_micro']:
            epoch_max = epoch + 1
            metrics_max_val = metrics['f1_micro']
            if args.best_model_path is not None:
                os.makedirs(args.best_model_path, exist_ok=True)
                # 包括模型参数及额外状态信息
                checkpoint = {'epoch': epoch + 1,
                              'metrics':metrics_max,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                torch.save(checkpoint, args.best_model_path + "check.pth")
            # # 加载完整的模型状态
            # checkpoint = torch.load(args.best_model_path)
        # 顺便看测试集
        #####################################################
    print()
    print("验证集最好epoch:",epoch_max)
    print(f"epoch: {epoch_max_test}, 最好的threshould:{best_thre:.2f}, ",",".join(f"{k}: {v:.4f}" for k, v in metrics_max.items()))
    print()
    # ############################################################
    # # 存储
    # if metrics_max_val < metrics['f1_micro']:
    #     epoch_max = epoch + 1
    #     metrics_max_val = metrics['f1_micro']
    #     if args.best_model_path is not None:
    #         os.makedirs(args.best_model_path, exist_ok=True)
    #         # 包括模型参数及额外状态信息
    #         checkpoint = {'epoch': epoch + 1,
    #                       'metrics':metrics_max,
    #                       'model_state_dict': model.state_dict(),
    #                       'optimizer_state_dict': optimizer.state_dict()}
    #         torch.save(checkpoint, args.best_model_path + "check.pth")
    #     # # 加载完整的模型状态
    #     # checkpoint = torch.load(args.best_model_path)
    # # 顺便看测试集
    # #####################################################



if __name__ == "__main__":
    main()