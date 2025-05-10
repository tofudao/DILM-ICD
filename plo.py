import torch
import numpy as np

def get_test_f1(test_dataloader, model):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    metrics_max = None
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if model.name == "plm_model":
                outputs,att = model(**batch)
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
    for t in np.linspace(0.0001, 0.9999, 100):
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
        micro_f1_list.append(metrics['f1_micro'])
        macro_f1_list.append(metrics['f1_macro'])
        # if metrics['f1_micro'] > max_microf1 and not math.isnan(metrics_max['f1_micro']):
        #     max_microf1 = metrics['f1_micro']
        # if metrics['f1_macro'] > max_macrof1 and not math.isnan(metrics_max['f1_macro']):
        #     max_macrof1 = metrics['f1_macro']

    return micro_f1_list,macro_f1_list