from src.train_load import *
from src.args_parse_2 import *
from vocab_all import *
from title_embedding import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW,get_scheduler
from tqdm.autonotebook import tqdm
from evaluation import all_metrics
import random
import logging
import math
import torch
logger = logging.getLogger(__name__)
import datetime
from src.model_strim import *

# set the random seed if needed, disable by default
set_random_seed(random_seed=42)

def main():
    args = create_args_parser()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # print(args)
    raw_datasets=load_data(args.data_dir)
    print(raw_datasets["train"])
    vocab = Vocab(args,raw_datasets)
    label_to_id=vocab.label_to_id
    print(label_to_id)
    print(len(vocab.label_dict['train']))
    print(len(vocab.label_dict['valid']))
    print(len(vocab.label_dict['test']))

    #存储title_icd_embdding
    if args.train_title_emb:
        print("使用训练标签标题")
        train_emb_title(label_to_id, title_emb_mode="skipgram", title_fea_size=10)
    # ##################################################################
    # # if args.is_trans:
    # if True:
    #     # lab_file = '../data/mimic3/lable_title_768.pkl'
    #     lab_file=args.label_titlefile
    #     with open(lab_file, 'rb') as file:
    #         labDescVec = pickle.load(file)
    #         #type(labDescVec) <class 'dict'>
    #         print("labDescVec的key",labDescVec.keys())
    #         labDescVec = labDescVec['id2descVec']
    #         # label_to_id_full=labDescVec['label_to_id']
    #     labDescVec = torch.nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32).unsqueeze(0),requires_grad=True)
    #     # labDescVec = torch.nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32).unsqueeze(0))
    # else:
    #     labDescVec=None
    # ########################################################################
    # train_emb_title(label_to_id, title_emb_mode="roberta-PM", title_fea_size=10)
    remove_columns = raw_datasets["train"].column_names

    print(remove_columns)

    def getitem(examples):

        result = dict()
        input_list = []
        label_list = []

        for text in examples["Text"]:
            text_list = text.split()
            if len(text_list) > args.max_seq_length:
                text_list=text_list[:args.max_seq_length]
            # elif len(text_list) < args.max_seq_length:
            #     for i in range(args.max_seq_length-len(text_list)):
            #         text_list.append('_PAD')
            print(text_list)

            input_list.append([vocab.word2index[word] for word in text_list])

        for labels in examples["Full_Labels"]:
            label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split('|')])

        result["input_ids"] = input_list
        result["label_ids"] = label_list

        return result
    def data_collator_train(features):
        # print("batch大小", len(features))
        lenth_list=[len(feature['input_ids']) for feature in features]
        length_list = []
        input_list=[]
        label_list=[]
        batch=dict()
        sorted_indices = sorted(range(len(features)), key=lambda i:len(features[i]['input_ids']), reverse=True)

        for i in sorted_indices:  # 数据增强
            len_fea = int(len(features[i]['input_ids']))
            length_list.append(len_fea)
            # ==========================================================================================================
            if args.dataEnhance:
                if random.random() < args.dataEnhanceRatio / 2:  # 随机排列
                    features[i]['input_ids'][:len_fea] = torch.tensor(features[i]['input_ids'][:len_fea])[np.random.permutation(len_fea)].tolist()
                if random.random() < args.dataEnhanceRatio:  # 逆置
                    features[i]['input_ids'][:len_fea] = torch.tensor(features[i]['input_ids'][:len_fea])[range(len_fea)[::-1]].tolist()
            # ==========================================================================================================
            input_list.append(torch.tensor(features[i]['input_ids']))
            label_tensor=torch.zeros(vocab.label_num)
            for j in features[i]['label_ids']:
                label_tensor[j]=1
            label_list.append(label_tensor)
        #pad一下
        padded_batch = pad_sequence(input_list, batch_first=True)
        batch['input_ids']=torch.LongTensor(padded_batch)
        batch['label_ids']=torch.tensor([label_bat.tolist() for label_bat in label_list])
        batch['length_input']=torch.tensor(length_list)
        # batch["labDescVec"] = labDescVec
        return batch

    def data_collator(features):

        lenth_list=[len(feature['input_ids']) for feature in features]
        length_list = []
        input_list=[]
        label_list=[]
        batch=dict()
        sorted_indices = sorted(range(len(features)), key=lambda i:len(features[i]['input_ids']), reverse=True)

        for i in sorted_indices:  # 数据增强
            len_fea = int(len(features[i]['input_ids']))
            length_list.append(len_fea)
            # # ==========================================================================================================
            # if args.dataEnhance:
            #     if random.random() < args.dataEnhanceRatio / 2:  # 随机排列
            #         features[i]['input_ids'][:len_fea] = torch.tensor(features[i]['input_ids'][:len_fea])[np.random.permutation(len_fea)].tolist()
            #     if random.random() < args.dataEnhanceRatio:  # 逆置
            #         features[i]['input_ids'][:len_fea] = torch.tensor(features[i]['input_ids'][:len_fea])[range(len_fea)[::-1]].tolist()
            # # ==========================================================================================================
            input_list.append(torch.tensor(features[i]['input_ids']))
            label_tensor=torch.zeros(vocab.label_num)
            for j in features[i]['label_ids']:
                label_tensor[j]=1
            label_list.append(label_tensor)
        #pad一下
        padded_batch = pad_sequence(input_list, batch_first=True)
        batch['input_ids']=torch.LongTensor(padded_batch)
        batch['label_ids']=torch.tensor([label_bat.tolist() for label_bat in label_list])
        batch['length_input']=torch.tensor(length_list)
        # batch["labDescVec"] = labDescVec
        return batch

    processed_datasets = raw_datasets.map(getitem, batched=True, remove_columns=remove_columns)# fn_kwargs={"vocab": vocab}
    # print(processed_datasets)
    # processed_datasets=dict()
    # processed_datasets["train"]=TextDataset(text_data=raw_datasets["train"],max_seq_length=args.max_seq_length,vocab=vocab,sort = True)
    # processed_datasets["valid"]=TextDataset(text_data=raw_datasets["valid"],max_seq_length=args.max_seq_length,vocab=vocab,sort = True)
    # processed_datasets["test"]=TextDataset(text_data=raw_datasets["test"],max_seq_length=args.max_seq_length,vocab=vocab,sort = True)

    train_dataloader = DataLoader(processed_datasets["train"], shuffle=True,collate_fn=data_collator_train, batch_size=args.batch_size)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    # model = RNN(args,vocab)
    model=RNN_Strim(args,vocab)
    if args.optimiser.lower() == "adamw":
        betas = (0.9, 0.999)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, betas=betas, weight_decay=args.weight_decay)

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
        print("itersPerEpoch",itersPerEpoch)
        # epoch = args.num_train_epochs
        epoch = 16
        # warmupEpochs = args.warmup
        warmupEpochs = 2
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                       num_training_steps=epoch * itersPerEpoch)

    lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=2000,
            num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
        )
    print("optimizer",optimizer)
    print("lr_scheduler",lr_scheduler)

    criterions = nn.BCEWithLogitsLoss()
    print("optimizer",optimizer)
    print("lr_scheduler",lr_scheduler)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    batch_id = 0
    metrics_max = None
    for epoch in tqdm(range(args.n_epoch)):
        print("第%d轮"%(epoch+1))
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        model.train()
        optimizer.zero_grad()
        losses=[]
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch_id += 1
            # batch['length_input']= batch['length_input'].to('cpu')
            output = model(batch)
            loss = criterions(output, batch['label_ids'])
            # loss = n_label * loss
            # loss=loss / n_label
            losses.append(loss.item())
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=epoch_loss / batch_id)
        # for step, batch in enumerate(train_dataloader):
        #     batch_id += 1
        #     batch['length_input']= batch['length_input'].to('cpu')
        #     with autocast():
        #         output = model(batch)
        #         loss = criterions(output, batch['label_ids'])
        #         # loss =criterions(output.view(-1, vocab.label_num),label_batch.view(-1, vocab.label_num))
        #     scaler.scale(loss).backward()
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
        for step, batch in tqdm(enumerate(eval_dataloader)):

            batch['length_input'] = batch['length_input'].to('cpu')
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
        logger.info(f"epoch {epoch} finished")
        logger.info(f"metrics: {metrics}")
        print("学习率lr{}:".format(optimizer.param_groups[0]["lr"]),"loss: ",np.mean(losses).item())
        print(f"验证集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, "
              f"auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, "
              f"prec_at_8: {metrics['prec_at_8']:.4f}")
        print(f"prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")
        # print("f1_micro: ", metrics['f1_micro'], "f1_macro: ", metrics['f1_macro'], "auc_micro: ", metrics['auc_micro'],
        #       "auc_macro: ", metrics['auc_macro'])
        # print("prec_at_8: ", metrics['prec_at_8'])
        metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)
        ##########################################
        if metrics_max is None:
            metrics_max = metrics_max_2
            best_thre = best_thre_2
        else:
            if metrics_max['f1_micro'] < metrics_max_2['f1_micro']:
                metrics_max = metrics_max_2
                best_thre = best_thre_2
    print("最好的阈值", best_thre, "最大的值：", metrics_max)
    #     print("验证集结果",metrics)
    #     ##########################################
    #     if metrics_max is None:
    #         metrics_max = metrics
    #     else:
    #         if metrics_max['f1_micro'] < metrics['f1_micro']:
    #             metrics_max = metrics
    #     ans_test(test_dataloader, model, args.code_50)
    # print("最大的值：",metrics_max)
if __name__ == "__main__":
    main()