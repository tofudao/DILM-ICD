from pathlib import Path
from collections import defaultdict
import pyarrow as pa
import pyarrow.feather
import vaex
import pyarrow.compute as pc



def get_code_system2code_counts(df, code_systems):
    code_system2code_counts = defaultdict(dict)
    for col in code_systems:
        codes = df[col].values.flatten().value_counts().to_pylist()
        code_system2code_counts[col] = {
            code["values"]: code["counts"] for code in codes
        }
    return code_system2code_counts

##########################
ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"
#####################################
def get_data_clean(direct=None,data_filename=None,split_filename=None,code_column_names=None):
    # direct="D:/study/毕业大论文/code/MIMIC4/medical-coding-reproducibility-main/files/data/mimiciii_clean"
    direct="../data/mimicdata/mimiciii_clean"
    data_filename="mimiciii_clean.feather"
    split_filename="mimiciii_clean_splits.feather"
    code_column_names=["icd9_diag","icd9_proc"]
    dir = Path(direct)
    with vaex.cache.memory_infinite():  # type: ignore
        read_a=pyarrow.feather.read_table(dir / data_filename,
                                   columns=[ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN, "num_words",
                                            "num_targets", ] + code_column_names, )
        df = vaex.from_arrow_table(read_a)
        # print("read_a",read_a)
        # print("这里是df",df)
        read_b=pyarrow.feather.read_table(dir / split_filename,)
        splits = vaex.from_arrow_table(read_b)
        # print("read_b",read_b)
        # print("splits::")
        # print(splits)
        df = df.join(splits, on=ID_COLUMN, how="inner")
        code_system2code_counts = get_code_system2code_counts(
            df, code_column_names
        )
        # print("code_system2code_counts",code_system2code_counts)
        print("code_system2code_counts",len(code_system2code_counts['icd9_diag']))
        print("code_system2code_counts",len(code_system2code_counts['icd9_proc']))
        schema = pa.schema(
            [
                pa.field(ID_COLUMN, pa.int64()),
                pa.field(TEXT_COLUMN, pa.large_utf8()),
                pa.field(TARGET_COLUMN, pa.list_(pa.large_string())),
                pa.field("split", pa.large_string()),
                pa.field("num_words", pa.int64()),
                pa.field("num_targets", pa.int64()),
            ]
        )
        # print("schema",schema)
        data=dict()
        df=df[[ID_COLUMN,TEXT_COLUMN,TARGET_COLUMN,"split","num_words","num_targets",]].to_arrow_table().cast(schema)
        data["train"]=df.filter(pc.field("split") == "train")
        data["val"]=df.filter(pc.field("split") == "val")
        data["test"]=df.filter(pc.field("split") == "test")
        print(type(df))

        return data,code_system2code_counts
        # print(data["train"]['target'])
        # print(data["val"]['target'])
        # print(data["test"]['target'])
        # print(len(data["train"]['split']))
        # print(len(data["val"]['split']))
        # print(len(data["test"]['split']))

# def load_data_new(data_dir):
#     train_path=data_dir+"train.csv"
#     valid_path=data_dir+"valid.csv"
#     test_path=data_dir+"test.csv"
#     data_files = {}
#     data_files["train"] = train_path
#     data_files["valid"] = valid_path
#     data_files["test"] = test_path
#     raw_datasets = load_dataset("csv", data_files=data_files)
#     return raw_datasets

def save_data(data,data_dir):
    train_path=data_dir+"train.csv"
    valid_path=data_dir+"valid.csv"
    test_path=data_dir+"test.csv"

    data["train"]=data["train"].to_pandas()
    data["valid"]=data["val"].to_pandas()
    data["test"]=data["test"].to_pandas()
    data["train"]['target'] = data["train"]['target'].apply(lambda x: ','.join(map(str, x)))
    data["valid"]['target'] = data["valid"]['target'].apply(lambda x: ','.join(map(str, x)))
    data["test"]['target'] = data["test"]['target'].apply(lambda x: ','.join(map(str, x)))
    len_train=int(len(data["train"])/2)
    data["train"]=data["train"][:len_train]
    print(data["train"])
    #################3
    data["train"].to_csv(train_path,index=False)  # 将 DataFrame 写入 CSV 文件
    data["valid"].to_csv(valid_path, index=False)  # 将 DataFrame 写入 CSV 文件
    data["test"].to_csv(test_path, index=False)  # 将 DataFrame 写入 CSV 文件


def main():
    data, code_system = get_data_clean(direct=None, data_filename=None, split_filename=None, code_column_names=None)
    data_dir="../data/mimicdata/mimiciii_clean_half/"
    save_data(data=data,data_dir=data_dir)
    # x = data["val"].to_pandas()
    # print(len(data["val"]["target"]))
    # print(type(data["val"]["target"]))
    # print(type(x))
    # print(type(x["_id"].values))
    # print(x["target"].values)
    # print(len(x["target"].values))

if __name__ == "__main__":
    main()
