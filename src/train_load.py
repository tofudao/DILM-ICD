import torch
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast,GradScaler
from src.evaluation import all_metrics
import numpy as np
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler=GradScaler()

class TextDataset(Dataset):

    def __init__(self,text_data,max_seq_length,vocab,sort = False,):
        super(TextDataset, self).__init__()
        self.max_seq_length=max_seq_length
        self.vocab = vocab
        indexed_data = []
        #训练集长度
        self.n_instances = len(text_data)
        self.n_total_tokens = 0

        n_label_level = len(text_data["Full_Labels"])
        self.label_count = dict()
        self.labels = set()
        for features in tqdm(text_data, desc="Processing data"):
            label_id_list=[]
            label_list =[label for label in features["Full_Labels"].strip().split('|')]
            for label in label_list:
                label_id = self.vocab.label_to_id[label]
                if label_id not in self.label_count:
                    self.label_count[label_id] = 1
                else:
                    self.label_count[label_id] += 1
                    self.labels.add(label_id)
                    label_id_list.append(label_id)
            word_seq = []

            records = features["Text"].split("\n\n")
            should_break = False
            for i in range(len(records)):
                record = records[i]
                sentences = record.split("\n")

                for sentence in sentences:
                    sent_words = sentence.strip().split()
                    if len(sent_words) == 0:
                        continue
                    # self.n_total_tokens += len(sent_words)
                    for word in sent_words:
                        word_idx = vocab.index_of_word(word)
                        word_seq.append(word_idx)
                        self.n_total_tokens += 1
                        if len(word_seq) >= self.max_seq_length > 0:
                            should_break = True
                            break
                    if should_break:
                        break

                if should_break:
                    break

            # after processing all records
            if len(word_seq) > 0:
                indexed_data.append((word_seq, label_id_list))

        if sort:
            self.indexed_data = sorted(indexed_data, key=lambda x: -len(x[0]))
        else:
            self.indexed_data =indexed_data
        self.labels = sorted(list(self.labels))

    def __getitem__(self, index):
        word_seq, label_list  = self.indexed_data[index]

#         label_id_list = [0] * self.vocab.label_num
#         for label in label_list:
#             label_id_list[label] = 1
        # label_id_list=np.asarray(label_id_list).astype(np.int32)
        result=dict()
        result["input_ids"] = word_seq
        result["label_ids"] = label_list
        return result

    def __len__(self):
        return len(self.indexed_data)


def ans_test(test_dataloader,model):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    metrics_max=None
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if model.name=="plm_model":
                outputs = model(**batch)
            elif model.name=="rnn_model":
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
#     print("测试结果")
#     logger.info(f"test finished")
#         logger.info(f"metrics: {metrics}")

    list_threshold=[0.3, 0.35, 0.4, 0.45, 0.5]

    best_thre=0
    for t in list_threshold:
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
        print(f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(f"prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")
        ##########################################
        if metrics_max is None:
            metrics_max=metrics
            best_thre=t
        else:
            if metrics_max['f1_micro']<metrics['f1_micro']:
                metrics_max=metrics
                best_thre=t
    return metrics_max,best_thre
##################################################


def ans_test_all(test_dataloader,model):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    metrics_max=None
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if model.name=="plm_model":
                outputs = model(**batch)
            elif model.name=="rnn_model":
                outputs = model(batch)
        preds_raw = outputs.sigmoid().cpu()
        all_preds_raw.extend(list(preds_raw))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))

    all_preds_raw = np.stack(all_preds_raw)
    all_labels = np.stack(all_labels)

    best_thre = 0
    for t in np.linspace(0.35, 0.55, 10):
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
        print(
            f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(f"prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")
        ##########################################
        if metrics_max is None:
            metrics_max = metrics
            best_thre = t
        else:
            if metrics_max['f1_micro'] < metrics['f1_micro']:
                metrics_max = metrics
                best_thre = t
    metrics=metrics_max

    print(
        f"测试集最大值: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    print(f"threshould:{best_thre:.4f}, prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")

    return metrics_max, best_thre

##################################################

import matplotlib.pyplot as plt
def get_test_f1(test_dataloader, model):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    metrics_max = None
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

    micro_f1_list=[]
    macro_f1_list=[]
    best_thre = 0
    for t in np.linspace(0, 1, 5):
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
        micro_f1_list.append(metrics['f1_micro'])
        macro_f1_list.append(metrics['f1_macro'])


    return micro_f1_list,macro_f1_list

def plot_f1(f1,name='micro F1'):
    lenth=len(f1)
    x=np.linspace(0, 1, lenth)
    # 绘制折线图
    plt.figure(figsize=(10, 5))  # 可以调整图形大小
    plt.plot(x, f1, marker='o')  # 使用圆圈标记数据点

    # 添加标题和轴标签

    plt.xlabel('决策边界')
    plt.ylabel(name)
    # 显示网格
    plt.grid(True)
    # 显示图表
    plt.show()

##################################################



def ans_test_all(test_dataloader,model):
    model.eval()
    all_preds_raw = []
    all_labels = []
    metrics_max=None
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if model.name=="plm_model":
                outputs = model(**batch)
            elif model.name=="rnn_model":
                outputs = model(batch)
        preds_raw = outputs.sigmoid().cpu()
        all_preds_raw.extend(list(preds_raw))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))

    all_preds_raw = np.stack(all_preds_raw)
    all_labels = np.stack(all_labels)

    best_thre = 0
    for t in np.linspace(0.35, 0.55, 10):
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
        print(
            f"测试集: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(f"prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")
        ##########################################
        if metrics_max is None:
            metrics_max = metrics
            best_thre = t
        else:
            if metrics_max['f1_micro'] < metrics['f1_micro']:
                metrics_max = metrics
                best_thre = t
    metrics=metrics_max

    print(
        f"测试集最大值: F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    print(f"threshould:{best_thre:.4f}, prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")

    return metrics_max, best_thre

##################################################
