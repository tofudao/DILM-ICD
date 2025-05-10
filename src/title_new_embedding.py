import pickle
from vocab_all import *
from src.args_parse_2 import *
import numpy as np
import pandas as pd

from process import *
from transformers import AutoTokenizer,RobertaModel
import torch.nn as nn

def reformat_icd(code: str, version: int, is_diag: bool) -> str:
    """format icd code depending on version"""
    if version == 9:
        return reformat_icd9(code, is_diag)
    elif version == 10:
        return reformat_icd10(code, is_diag)
    else:
        raise ValueError("version must be 9 or 10")


def reformat_icd10(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if not is_diag:
        return code
    return code[:3] + "." + code[3:]


def reformat_icd9(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code

# pd.set_option('display.max_columns', None)
def init_title_icd9(label2index):
    # tword2id, id2tword = {"<EOS>": 0}, ["<EOS>"]
    ICD_dic=set()
    mimicPath = "../data/mimicdata/icd_title/"
    dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    print(dIcdDiagnoses.head(5))
    for dic, code in dIcdDiagnoses['ICD9_CODE'].items():
        # print(code,type(code),dIcdDiagnoses.loc[dic]['ICD9_CODE'])
        dIcdDiagnoses.loc[dic, 'ICD9_CODE'] = reformat_icd(code=str(code), version=9,is_diag=True)
        ICD_dic.add(dIcdDiagnoses.loc[dic, 'ICD9_CODE'])
    print(dIcdDiagnoses.head(5))
    dIcdDiagnoses = dIcdDiagnoses.set_index('ICD9_CODE')
    dicdProcedures = pd.read_csv(os.path.join(mimicPath, 'D_ICD_PROCEDURES.csv'), dtype=str)
    dicdProcedures['ICD9_CODE'] = dicdProcedures['ICD9_CODE'].astype(str)
    for dic, code in dicdProcedures['ICD9_CODE'].items():
        dicdProcedures.loc[dic, 'ICD9_CODE'] = reformat_icd(code=str(code), version=9,is_diag= False)
        ICD_dic.add(dicdProcedures.loc[dic, 'ICD9_CODE'])
    dicdProcedures = dicdProcedures.set_index('ICD9_CODE')
    icdTitles = pd.concat([dIcdDiagnoses, dicdProcedures])

    title_icd=dict()
    titles = []

    print("type(self.label2index)", type(label2index))
    print("len(self.label2index)", len(label2index))
    # print("self.all_labels", self.all_labels)
    # print("self.label2index[0]", label2index[0])
    cnt = 1
    count = 0
    xx = 0
    for icd in label2index:
        try:
            # 标签长描述和短描述拼接
            desc = (icd+" : "+icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower()
            xx += 1
        except:
            count += 1
            flag=False
            for i in range(10):
                if '.' not in icd:
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
    #不存在标题的个数 87存在标题的个数 3594
    print("不存在标题的个数", count)
    print("存在标题的个数", xx)
    print("标题描述中独立词的个数", cnt)
    return title_icd
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
    len_title=(len(title))
    tword2vec = np.zeros((len_title, title_fea_size), dtype=np.float32)
    for i in range(len_title):
        tword2vec[i] = model.wv[id_to_title[i]]
    ##############################################3
    #path="../data/mimicdata/mimiciii_clean/"
    path='../data/mimicdata/mimiciii_clean/n_clean_title_%s_d_%d.pkl'%(method,title_fea_size)
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

    return id2descVec




def train_emb_title(label2index,title_emb_mode,title_fea_size=None, model_name_or_path=None,
                    is_50=False,label_to_id_50=None):
    # title_icd = init_title(label2index)
    title_icd = init_title_icd9(label2index)

    if title_emb_mode == 'skipgram' or title_emb_mode == 'cbow':
        id2descVec=get_emb_word2vec(title_icd, label2index, model_name_or_path=model_name_or_path,
                                    method=title_emb_mode, title_fea_size=title_fea_size)
        ############
        #存储full
        data = {'id2descVec': id2descVec}
        data['label_to_id'] = label2index
        with open('../data/mimicdata/mimiciii_clean/split_lable_title_%s_full_%d.pkl' % (title_emb_mode, title_fea_size), 'wb') as file:
            pickle.dump(data, file)
        # file_path = "full_2.txt"
        # # 打开或创建一个文件，并将其设置为写入模式（'w'）
        # with open(file_path, 'w') as output_file:
        #     # 使用print()函数将内容打印到文件中，指定file参数为目标文件对象
        #     print(torch.tensor(id2descVec), file=output_file)
    elif title_emb_mode=='roberta-PM':
        model_name_or_path="../models/RoBERTa-base-PM-M3-Voc-distill-align-hf/"
        id2descVec=get_emb_roberta(title_icd, label2index, model_name_or_path)
        print(len(id2descVec))

        data = {'id2descVec':id2descVec}
        data['label_to_id']=label2index
        # print(data)
        # 将Python对象序列化为二进制数据
        ###############################

        with open('../data/mimicdata/mimiciii_clean/split_lable_title_RoBERTa_full_768.pkl', 'wb') as file:
            pickle.dump(data, file)


from vocab_new import *
def emb_new_data():
    # from src.get_new_data import *
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
    # emb_data()
    emb_new_data()


