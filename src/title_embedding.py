import pickle
from vocab_all import *
from src.args_parse_2 import *
import numpy as np
import pandas as pd

from process import *
from transformers import AutoTokenizer,RobertaModel
import torch.nn as nn


# pd.set_option('display.max_columns', None)
def init_title(label2index):
    # tword2id, id2tword = {"<EOS>": 0}, ["<EOS>"]
    ICD_dic=set()
    mimicPath = "../data/mimicdata/icd_title/"
    dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    print(dIcdDiagnoses.head(5))
    for dic, code in dIcdDiagnoses['ICD9_CODE'].items():
        # print(code,type(code),dIcdDiagnoses.loc[dic]['ICD9_CODE'])
        dIcdDiagnoses.loc[dic, 'ICD9_CODE'] = reformat(code, True)
        ICD_dic.add(dIcdDiagnoses.loc[dic, 'ICD9_CODE'])
    print(dIcdDiagnoses.head(5))
    dIcdDiagnoses = dIcdDiagnoses.set_index('ICD9_CODE')
    dicdProcedures = pd.read_csv(os.path.join(mimicPath, 'D_ICD_PROCEDURES.csv'), dtype=str)
    dicdProcedures['ICD9_CODE'] = dicdProcedures['ICD9_CODE'].astype(str)
    for dic, code in dicdProcedures['ICD9_CODE'].items():
        dicdProcedures.loc[dic, 'ICD9_CODE'] = reformat(str(code), False)
        ICD_dic.add(dicdProcedures.loc[dic, 'ICD9_CODE'])

    dicdProcedures = dicdProcedures.set_index('ICD9_CODE')
    icdTitles = pd.concat([dIcdDiagnoses, dicdProcedures])

    title_icd=dict()
    titles = []
    # id2icd列表记录了每一个icd
    ##### type(self.label2index) <class 'list'>
    ##### len(self.label2index) 1
    ####self.label2index[0] {'E030': 0,

    #type(self.label2index) <class 'dict'>
    print("type(self.label2index)", type(label2index))
    print("len(self.label2index)", len(label2index))
    # print("self.all_labels", self.all_labels)
    # print("self.label2index[0]", label2index[0])
    cnt = 1
    count = 0
    xx = 0

    #################################################
    # tword2id, id2tword = {"<EOS>": 0}, ["<EOS>"]
    # for icd in label2index:
    #     try:
    #         # 标签长描述和短描述拼接
    #         desc = (icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower().split()
    #         desc2=(icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower()
    #         xx += 1
    #     except:
    #         count += 1
    #         desc = " <:> ".split()
    #         desc2=" <:> "
    #     titles.append(desc + ["<EOS>"])
    #
    #     title_icd[icd] = desc2+" <EOS> "
    #     # print('desc[1]:',desc[1])
    #
    #     # 从文本标题长和短描述中得到所有标题的所有词，
    #     for w in desc:
    #         if w not in tword2id:
    #             tword2id[w] = cnt
    #             id2tword.append(w)
    #             cnt += 1
    # print("数量",cnt)
    # titleLen = [len(i) for i in titles]
    # # print(titleLen)
    # titleMaxLen = 25
    # print(titleMaxLen)
    # tokenizedTitle=[]
    # # for t in titles:
    # #     if len(t) > titleMaxLen:
    # #
    # #         tokenizedTitle.append([tword2id[w] for w in t[:]])
    # #     else:
    # #         k=[tword2id[w] for w in t]+ [0] * (titleMaxLen - len(t))
    # #         tokenizedTitle.append(k)
    # # tokenizedTitle=np.array(tokenizedTitle,dtype='int32')
    # tokenizedTitle = np.array(
    #     [[tword2id[w] for w in t[:titleMaxLen]] + [0] * (titleMaxLen - len(t[:titleMaxLen])) for t in titles], dtype='int32')
    # # tokenizedTitle = np.array(
    # #     [[tword2id[w] for w in t] for t in titles], dtype='int32')
    # return title_icd,tokenizedTitle
    ####################################################################
    for icd in label2index:
        try:
            # 标签长描述和短描述拼接
            desc = (icd+" : "+icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower()
            xx += 1
        except:
            count += 1
            flag=False
            for i in range(10):
                if '.' not in icd.split('_')[1]:
                    icd_fake = icd + '.'+str(i)
                else:
                    icd_fake=icd+str(i)
                if icd_fake in ICD_dic:
                    flag=True
                    desc = (icd+" : "+icdTitles.loc[icd_fake]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd_fake]['LONG_TITLE']).lower()
                    break
            if not flag:
                desc = (icd+ " <:> ").lower()
                print(icd_fake)
            # print(icd)
            # print(icd_fake)
            # desc = " <:> ".split()
        # title_temp=desc + ["<EOS>"]
        desc=desc+" <EOS> "
        titles.append(desc)
        title_icd[icd]=desc
    print("不存在标题的个数", count)
    print("存在标题的个数", xx)
    print("标题描述中独立词的个数", cnt)
    return title_icd,None
    #############################################################################


def get_emb_word2vec(title_icd,label2index,model_name_or_path=None,method="skipgram",title_fea_size=128,tokenizer=None):
    id2descVec = np.zeros((len(label2index), title_fea_size), dtype='float32')

    title_list=[]
    title_count = Counter()
    x=0
    for icd,desc in title_icd.items():
        icd_title_list =desc.split()
        if x ==0:
            print(icd_title_list)
            x+=1
        title_list.append(icd_title_list)
        title_count.update(icd_title_list)
    # print(title_list)
    if model_name_or_path is None:
        if method=="skipgram":
            model = Word2Vec(title_list, min_count=0, window=5, vector_size=title_fea_size, workers=8, sg=1, epochs=10)
        elif method=="cwob":
            model = Word2Vec(title_list, min_count=0, window=5, vector_size=title_fea_size, workers=8, sg=0, epochs=10)
    else:
        # if os.path.exists(model_name_or_path):
        with open(model_name_or_path, 'rb') as f:
            titleEmbedding = pickle.load(f)
        #titleEmbedding.shape: (9843, 512)
        print('titleEmbedding.shape:', titleEmbedding.shape)
        #####################################
        title_emb_layer=nn.Embedding.from_pretrained(torch.tensor(titleEmbedding, dtype=torch.float32), freeze=False)
        #tokenizedTitle torch.Size([8929, 36])
        tokenizedTitle = torch.LongTensor(tokenizer)
        title_vec = title_emb_layer(tokenizedTitle).detach().data.cpu().numpy().mean(axis=1)
        print("tokenizedTitle",tokenizedTitle.shape)
        #title_vec (8929, 512)
        print("title_vec",title_vec.shape)
        for i in range(len(title_vec)):
            id2descVec[i]=title_vec[i]
        # for icd in label2index:
        #     tokenized_title = tokenizer
        #     tokenizedTitle = torch.LongTensor(tokenized_title)
        #     title_vec = title_emb_layer(tokenizedTitle).detach().data.cpu().numpy().mean(axis=0)
        #     title_vec = torch.tensor(title_vec).unsqueeze(0)
        #     id2descVec[label2index[icd]] = title_vec
            # print(id2descVec.shape)
        return id2descVec
    print("搞不到")
    title = sorted(title_count.keys())
    title_to_id = {v: i for i, v in enumerate(title)}
    id_to_title={i: v for i, v in enumerate(title)}
    print("title_to_id[EOS]",title_to_id["<EOS>"])
    if True:
        len_title=(len(title))
        tword2vec = np.zeros((len_title, title_fea_size), dtype=np.float32)
        for i in range(len_title):
            tword2vec[i] = model.wv[id_to_title[i]]
        ##############################################3
        path='../data/embeddings/n_title_%s_d_%d.pkl'%(method,title_fea_size)
        with open(path, 'wb') as f:
            pickle.dump(tword2vec, f, protocol=4)
        #####################################################
        title_emb_layer=nn.Embedding.from_pretrained(torch.tensor(tword2vec, dtype=torch.float32), freeze=False)
        for icd in label2index:
            tokenized_title = np.array([title_to_id[token] for token in title_icd[icd].split()],dtype='int32')
            tokenizedTitle = torch.LongTensor(tokenized_title)

            title_vec = title_emb_layer(tokenizedTitle).detach().data.cpu().numpy().mean(axis=0)
            if x ==1:
                print(tokenizedTitle.shape)
                print(title_vec.shape)
                x+=1
            id2descVec[label2index[icd]] = title_vec
            # print(id2descVec.shape)
        return id2descVec
        # title_vec = nn.Parameter(torch.tensor(title_vec, dtype=torch.float32, requires_grad=False))

    #本来应该存储这个，但是我们直接计算标签的值
    # word2vec = np.zeros((len(title), title_fea_size), dtype=np.float32)
    # for i,v in enumerate(title):
    #     word2vec[i]=model.wv[v]
    for icd in label2index:
        tokenized_title=np.array([model.wv[token] for token in title_icd[icd].split()])
        print(tokenized_title.shape)
        tmp=tokenized_title.mean(axis=0)
        id2descVec[label2index[icd]] = tmp

    # print(id2descVec.shape)
    # return id2descVec


def get_emb_roberta(title_icd,label2index,model_name_or_path):
    # id2descVec=[]
    id2descVec = np.zeros((len(label2index), 768), dtype='float32')
    # config = AutoConfig.from_pretrained(model_name_or_path)

    # config.model_mode = 'roberta'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = RobertaPreTrainedModel.from_pretrained(model_name_or_path).eval()
    model = RobertaModel.from_pretrained(model_name_or_path).eval().to('cuda:0')
    # model = RobertaModel.from_pretrained(model_name_or_path,from_tf = bool(".ckpt" in model_name_or_path),config=config)
    for icd,desc in title_icd.items():
        # model = RobertaModel.from_pretrained(
        #             'trainMLM/model/roberta_large/',
        #         ).eval().to('cuda:0')
        # print("label2index[icd]",label2index[icd])
        # titleLen = [len(i) for i in titles]
        titleLen = len(desc)
        # print(titleLen)
        # print(type(desc))
        # print(desc)
        # tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True,
        #           add_special_tokens="cls" not in args.model_mode)
        # print("desc",desc)
        inputs = tokenizer(desc,padding=False,max_length=titleLen,truncation=True,)
        # print("input",inputs)
        tokens = tokenizer.tokenize(desc)
        # print(tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to('cuda:0')
        # print(type(input_ids))
        # print(input_ids)
        # inputs['input_ids']=input_ids.to("cuda:0")
        # print(inputs)
        # tmp = model(**inputs)[0].detach().data.cpu().numpy().mean(axis=1)
        tmp=model(input_ids)
        # print("tmp=tmp[0].,",tmp[0].size())
        tmp=tmp[0].detach().data.cpu().numpy().mean(axis=1)

        # print(type(tmp[0]))
        # print(len(tmp[0]))
        # print(tmp[1].size())
        id2descVec[label2index[icd]] = tmp

    # print(type(id2descVec))
    # print(id2descVec)

    # titleLen = [len(i) for i in titles]
    # titleMaxLen = max(titleLen)
    # twordNum = cnt
    # tokenizedTitle = np.array(
    #     [[tword2id[w] for w in t] + [0] * (titleMaxLen - len(t)) for t in titles], dtype='int32')
    # totalSampleNum = len(tokenizedTitle)
    return id2descVec




def train_emb_title(label2index,title_emb_mode,title_fea_size=None, model_name_or_path=None,
                    is_50=False,label_to_id_50=None):
    # title_icd = init_title(label2index)
    title_icd,tokenizer = init_title(label2index)

    if title_emb_mode == 'skipgram' or title_emb_mode == 'cbow':
        id2descVec=get_emb_word2vec(title_icd, label2index, model_name_or_path=model_name_or_path,
                                    method=title_emb_mode, title_fea_size=title_fea_size,tokenizer=tokenizer)
        ############
        #存储full
        data = {'id2descVec': id2descVec}
        data['label_to_id'] = label2index
        with open('../data/embeddings/lable_title_%s_full_%d.pkl' % (title_emb_mode, title_fea_size), 'wb') as file:
            pickle.dump(data, file)
        file_path = "full_2.txt"
        # 打开或创建一个文件，并将其设置为写入模式（'w'）
        with open(file_path, 'w') as output_file:
            # 使用print()函数将内容打印到文件中，指定file参数为目标文件对象
            print(torch.tensor(id2descVec), file=output_file)
        ############
        #存储50
        if is_50:
            id2descVec_50 = np.zeros((50, title_fea_size), dtype='float32')
            print("len",len(label_to_id_50))
            for icd in label_to_id_50:
                # print(icd)
                # print(label2index[icd])
                id2descVec_50[label_to_id_50[icd]]=id2descVec[label2index[icd]]
            data = {'id2descVec':id2descVec_50}
            data['label_to_id']=label2index
            with open('../data/embeddings/lable_title_%s_50_%d.pkl'%(title_emb_mode,title_fea_size), 'wb') as file:
                pickle.dump(data, file)
            # 假设您有一个变量需要写入文件
            file_path = "50_2.txt"
            # 打开或创建一个文件，并将其设置为写入模式（'w'）
            with open(file_path, 'w') as output_file:
                # 使用print()函数将内容打印到文件中，指定file参数为目标文件对象
                print(torch.tensor(id2descVec_50), file=output_file)
        # else:
        #     data = {'id2descVec':id2descVec}
        #     data['label_to_id']=label2index
        #     if len(label2index)==50:
        #         with open('../data/embeddings/lable_title_%s_50_%d.pkl'%(title_emb_mode,title_fea_size), 'wb') as file:
        #             pickle.dump(data, file)
        #     else:
        #         with open('../data/embeddings/lable_title_%s_full_%d.pkl'%(title_emb_mode,title_fea_size), 'wb') as file:
        #             pickle.dump(data, file)
    elif title_emb_mode=='roberta-PM':
        model_name_or_path="../models/RoBERTa-base-PM-M3-Voc-distill-align-hf/"
        id2descVec=get_emb_roberta(title_icd, label2index, model_name_or_path)
        print(len(id2descVec))

        data = {'id2descVec':id2descVec}
        data['label_to_id']=label2index
        # print(data)
        # 将Python对象序列化为二进制数据
        ###############################
        if len(label2index)==50:
            with open('../data/embeddings/lable_title_RoBERTa_50_768.pkl', 'wb') as file:
                pickle.dump(data, file)
        else:
            with open('../data/embeddings/lable_title_RoBERTa_full_768.pkl', 'wb') as file:
                pickle.dump(data, file)

def emb_data():
    # def set_random_seed(random_seed):
    #     torch.manual_seed(random_seed)
    #     torch.cuda.manual_seed_all(random_seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # set_random_seed(random_seed=42)
    # random.seed(42)
    # np.random.seed(42)
    title_emb_mode = "roberta-PM"
    if title_emb_mode == "skipgram":
        args = create_args_parser()
        args.data_dir = "../data/mimicdata/mimic3/new/full/"
        raw_datasets = load_data(args.data_dir)
        print(raw_datasets)
        vocab = Vocab(args, raw_datasets)
        label_to_id = vocab.label_to_id
        is_50 = True
        if is_50 == True:
            args.data_dir = "../data/mimicdata/mimic3/new/50/"
            raw_datasets_50 = load_data(args.data_dir)
            vocab_50 = Vocab(args, raw_datasets_50)
            label_to_id_50 = vocab_50.label_to_id
            # model_name_or_path="../data/embeddings/title_skipgram_d512.pkl"
            train_emb_title(label_to_id, title_emb_mode="skipgram", title_fea_size=768, is_50=is_50,
                            label_to_id_50=label_to_id_50)
        else:
            train_emb_title(label_to_id, title_emb_mode="skipgram", title_fea_size=1024)
    elif title_emb_mode == "roberta-PM":
        args = create_args_parser()
        is_50 = True
        if is_50 == True:
            args.data_dir = "../data/mimicdata/mimic3/new/50/"
        else:
            args.data_dir = "../data/mimicdata/mimic3/new/full/"
        raw_datasets = load_data(args.data_dir)
        print(raw_datasets)
        vocab = Vocab(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode='roberta-PM')

# # 从二进制数据反序列化为Python对象
# with open('data.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

from vocab_new import *
def emb_new_data():

    title_emb_mode = "skipgram"
    if title_emb_mode == "skipgram":
        args = create_args_parser()
        args.data_dir = "../data/mimicdata/mimiciii_clean/"
        raw_datasets = load_data(args.data_dir)
        # data_pa, code_system = get_data_clean(direct=None, data_filename=None, split_filename=None,
        #                                       code_column_names=None)
        print(raw_datasets)
        vocab = Vocab_new(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode="skipgram", title_fea_size=1024)

    elif title_emb_mode == "roberta-PM":
        args = create_args_parser()
        args.data_dir = "../data/mimicdata/mimiciii_clean/"
        raw_datasets = load_data(args.data_dir)
        print(raw_datasets)
        vocab = Vocab_new(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode='roberta-PM')


if __name__ == "__main__":
    emb_data()
    # emb_new_data()


