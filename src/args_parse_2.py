import argparse

# from src.models.rnn import *

def create_args_parser():
    parser = argparse.ArgumentParser()
    #============================================================================
    parser.add_argument("--tok_stm", type=int, default=512, help="token影藏参数")
    parser.add_argument("--label_stm", type=int, default=256, help="token影藏参数")

    #===================================
    parser.add_argument('--data_dir', type=str, default="../data/mimicdata/mimic3/new/50/")
    parser.add_argument('--data_new_dir', type=str, default="../data/mimicdata/mimiciii_clean/")
    parser.add_argument('--code_50', type=int, choices=[0, 1], default=1,help="是否是50的")
    parser.add_argument('--label_titlefile', type=str, default="../data/embeddings/lable_title_skipgram_50_10.pkl")
    parser.add_argument('--train_title_emb', action="store_true", help="是否进行训练标签的emb")

    parser.add_argument("--dataEnhance", action="store_true", help="是否使用Transformer")
    parser.add_argument("--dataEnhanceRatio", type=float, default=0.2, help="数据增强率")

    parser.add_argument("--is_trans", action="store_true", help="是否使用Transformer")
    parser.add_argument("--multiNum", type=int, default=4, help="Transformer的头数")
    parser.add_argument("--dk", type=int, default=128, help="Transformer的影藏参数")


    parser.add_argument("--lr_cosine", action="store_true", help="是否使用cosine")
    parser.add_argument("--warmup", type=int, default=2, help="预热")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--hidden_size", type=int, default=5, help="The size of the hidden layer")


    parser.add_argument('--iter_layer', type=int, default=1, help="迭代的层数")
    parser.add_argument('--filterSize', type=int, default=64)
    parser.add_argument('--reshape_size', type=int, default=256)

    # parser.add_argument("--problem_name", type=str, default="mimic-iii_single_50", required=False,
    #                     help="The problem name is used to load the configuration from config.json")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--n_epoch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping")

    parser.add_argument("--optimiser", type=str, choices=["adagrad", "adam", "sgd", "adadelta", "adamw"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--use_lr_scheduler", type=int, choices=[0, 1], default=1,
                        help="Use lr scheduler to reduce the learning rate during training")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.9,
                        help="Reduce the learning rate by the scheduler factor")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5,
                        help="The lr scheduler patience")

    parser.add_argument("--level_projection_size", type=int, default=128)

    parser.add_argument("--main_metric", default="micro_f1",
                        help="the metric to be used for validation",
                        choices=["macro_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
                                 "micro_accuracy", "micro_precision", "micro_recall", "micro_f1", "micro_auc", "loss",
                                 "macro_P@1", "macro_P@5", "macro_P@8", "macro_P@10", "macro_P@15"])
    parser.add_argument("--metric_level", type=int, default=1,
                        help="The label level to be used for validation:"
                             "\n\tn: The n-th level if n >= 0 (started with 0)"
                             "\n\tif n > max_level, n is set to max_level"
                             "\n\tif n < 0, use the average of all label levels"
                             )

    parser.add_argument("--multilabel", default=1, type=int, choices=[0, 1])

    parser.add_argument("--shuffle_data", type=int, choices=[0, 1], default=1)

    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")

    parser.add_argument("--save_best_model", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_results", type=int, choices=[0, 1], default=1)
    parser.add_argument("--best_model_path", type=str, default=None)
    parser.add_argument("--save_results_on_train", action='store_true', default=False)
    parser.add_argument("--resume_training", action='store_true', default=False)

    parser.add_argument("--max_seq_length", type=int, default=10)
    # parser.add_argument("--min_seq_length", type=int, default=-1)
    parser.add_argument("--min_word_frequency", type=int, default=-1)

    # Embedding
    parser.add_argument("--embedding_mode", type=str, default="static",
                        choices=["static", "non_static"],
                        help="The mode to init embeddings:"
                             "\n\t1. rand: initialise the embedding randomly"
                             "\n\t2. static: using pretrained embeddings"
                             "\n\t2. non_static: using pretrained embeddings with fine tuning"
                             "\n\t2. multichannel: using both static and non-static modes")
    # parser.add_argument("--embedding_mode", type=str, default="word2vec",
    #                     help="Choose the embedding mode which can be fasttext, word2vec")
    # parser.add_argument("--embedding_mode", type=str, default="fasttext",
    #                     help="Choose the embedding mode which can be fasttext, word2vec")
    parser.add_argument('--embedding_size', type=int, default=100)
    #parser.add_argument("--embedding_file", type=str, default=None)
    parser.add_argument('--word_embedding_file', type=str, default='../data/embeddings/word2vec_sg0_100.model')
    # parser.add_argument("--embedding_file", type=str, default='data/embeddings/word2vec_sg0_100.model')

    # Attention

    parser.add_argument("--attention_mode", type=str, choices=["text_label", "label", "caml"], default="text_label")
    #parser.add_argument("--d_a", type=int, help="The dimension of the first dense layer for self attention", default=-1)
    parser.add_argument("--d_a", type=int,default=512, help="The dimension of the first dense layer for self attention")

    # parser.add_argument("--use_regularisation", action='store_true', default=False)
    # parser.add_argument("--penalisation_coeff", type=float, default=0.01)

    # sub_parsers = parser.add_subparsers()
    #
    # #_add_sub_parser_for_cnn(sub_parsers)
    # _add_sub_parser_for_rnn(sub_parsers)
    args = parser.parse_args()
    return args


# def _add_sub_parser_for_rnn(subparsers):
#     args = subparsers.add_parser("RNN")
#     #args.add_argument("--hidden_size", type=int, default=100, help="The size of the hidden layer")
#     args.add_argument("--hidden_size", type=int, default=512, help="The size of the hidden layer")
#     args.add_argument("--n_layers", type=int, default=1, help="The number of hidden layers")
#     args.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
#                       help="Whether or not using bidirectional connection")
#     args.set_defaults(model=RNN)
#     args.add_argument("--use_last_hidden_state", type=int, choices=[0, 1], default=0,
#                       help="Using the last hidden state or using the average of all hidden state")
#     args.add_argument("--rnn_model", type=str, choices=["GRU", "LSTM"], default="LSTM")



