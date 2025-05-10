import json
import numpy as np
import os
import sys

from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import datasets
def pre_micro_f1(yhat, y):
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    intersect=np.logical_and(yhat, y).sum(axis=0)
    prec = intersect/ (yhatmic.sum(axis=0) + 1e-10)
    rec = intersect / (ymic.sum(axis=0) + 1e-10)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def pre_macro_f1(yhat, y):
    prec =intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    rec = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)

    f1 = 2*(prec*rec)/(prec+rec+ 1e-10)
    return f1

#大到小
def find_theta(y, yhat_raw):
    sample,label=y.shape
    f1_map = {
        'max_macro_f1': np.zeros(label),
        "max_theta": np.ones(label) / 2
    }


    yhat = (yhat_raw > 0.5).astype(int)

    macro_f1_all = pre_macro_f1(yhat, y)
    micro_f1_all=pre_micro_f1(yhat, y)
    print("macro_f1_all",macro_f1_all.shape)
    print("micro_f1_all",micro_f1_all)

    # sorted_array, indices = np.sort(macro_f1_all)
    indices = np.argsort(-macro_f1_all)
    # sort_order_desc = np.argsort(-macro_f1_all)
    yhat_temp =yhat.copy()
    max_mif=micro_f1_all
    for i in range(label):
        temp_i=indices[i]
        for t in np.linspace(0, 1,51):
        # for t in np.interp(np.linspace(0, 1, 51), (0, 1), indices[i]):
            yhat_temp[temp_i] = (yhat_raw[temp_i] > t).astype(int)
            micro_f1_temp = pre_micro_f1(yhat_temp, y)
            if micro_f1_temp>max_mif:
                max_mif=micro_f1_temp
                f1_map['max_macro_f1'] = pre_macro_f1(yhat_temp, y)
                f1_map['max_theta'][temp_i]=t
                f1_map['max_micro_f1']=max_mif
    the_first_f1 = np.mean(f1_map['max_macro_f1'])
    print("the_first_f1", the_first_f1)
    print("max_mif", max_mif)
    return f1_map
#小到大
def find_theta_2(y, yhat_raw):
    sample,label=y.shape
    print(y.shape)
    f1_map = {
        'max_macro_f1': np.zeros(label),
        "max_theta": np.ones(label) / 2
    }
    yhat = (yhat_raw > 0.5).astype(int)

    macro_f1_all = pre_macro_f1(yhat, y)
    micro_f1_all=pre_micro_f1(yhat, y)
    print("macro_f1_all",macro_f1_all.shape)
    print("micro_f1_all",micro_f1_all)

    # sorted_array, indices = np.sort(macro_f1_all)
    indices = np.argsort(macro_f1_all)
    # sort_order_desc = np.argsort(-macro_f1_all)
    yhat_temp =yhat
    max_mif=micro_f1_all
    for i in range(label):
        temp_i=indices[i]
        for t in np.linspace(0, 1,51):
            yhat_temp[temp_i] = (yhat_raw[temp_i] > t).astype(int)
            micro_f1_temp = pre_micro_f1(yhat_temp, y)
            if micro_f1_temp>max_mif:
                max_mif=micro_f1_temp
                f1_map['max_macro_f1'] = pre_macro_f1(yhat_temp, y)
                f1_map['max_theta'][temp_i]=t
                f1_map['max_micro_f1']=max_mif
    the_first_f1 = np.mean(f1_map['max_macro_f1'])
    print("the_first_f1", the_first_f1)
    print("max_mif", max_mif)
    return f1_map

def find_theta(y, yhat_raw):
    sample,label=y.shape
    print(y.shape)
    macro_map=dict()
    macro_map['max_macro_f1']=np.zeros(label)
    macro_map['max_macro_precision']=np.zeros(label)
    macro_map['max_macro_recall']=np.zeros(label)
    macro_map["max_theta"]=np.zeros(label)

    macro_map["max_yhat"]=np.zeros(label)
    for t in np.linspace(0.1, 0.9, 91):
        yhat = (yhat_raw > t).astype(int)

        macro_precision = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
        macro_recall = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
        #(50,0)
        # print("safa ",macro_precision.shape)

        macro_f1 = 2*(macro_precision*macro_recall)/(macro_precision+macro_recall+1e-10)

        for i in range(label):
            if macro_f1[i] > macro_map['max_macro_f1'][i]:
                macro_map['max_macro_f1'][i] = macro_f1[i]
                macro_map['max_macro_precision'][i]=macro_precision[i]
                macro_map['max_macro_recall'][i]=macro_recall[i]
                macro_map['max_theta'][i]=t
                macro_map["max_yhat"][i]=yhat.sum(axis=0)[i]
        # # 使用 NumPy 的广播和比较操作来更新 max_f1 数组
        # max_macro_f1 = np.maximum(max_macro_f1, macro_f1)

    print("macro_map[max_macro_f1]",macro_map['max_macro_f1'])
    print("macro_map[max_theta]",macro_map["max_theta"])
    the_first_f1=np.mean(macro_map['max_macro_f1'])
    prec=np.mean(macro_map['max_macro_precision'])
    rec=np.mean(macro_map['max_macro_recall'])
    if prec + rec == 0:
        the_second_f1 = 0.
    else:
        the_second_f1 = 2*(prec*rec)/(prec+rec)

    print("the_first_f1",the_first_f1)
    print("the_second_f1",the_second_f1)

    #===========================================
    #micro-f1
    yhat_raw_temp=np.zeros((sample,label))
    for i in range(label):
        theta=macro_map['max_theta'][i]
        yhat_raw_temp[:,i] = (yhat_raw[:,i] > theta).astype(int)
    #88405.0
    ymic = y.ravel()
    #9312966.0
    yhatmic = yhat_raw_temp.ravel()
    #55920
    micro_precision=intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)
    micro_recall=intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.
    else:
        micro_f1 = 2*(micro_precision*micro_recall)/(micro_precision+micro_recall)
    print(micro_f1)
    #=======================================================================
    return macro_map,the_first_f1,the_second_f1,micro_f1



def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


import numpy as np

# 假设 precision 和 recall 都是长度为 50 的 NumPy 数组
y = np.random.rand(10,50)  # 这里我们用随机数模拟精度值
yhat_raw = np.random.rand(10,50)      # 这里我们用随机数模拟召回率值
all_metrics(y, yhat_raw)
# # 计算 F1 值
# f1_scores = 2 * (precision * recall) / (precision + recall)
#
# # 打印 F1 值数组
# print(f1_scores)