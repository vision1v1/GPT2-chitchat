from transformers import BertTokenizerFast
import argparse
import pickle
from tqdm import tqdm
import logging
import numpy as np
from util import create_logger

def preprocess():
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--log_path', default='data/preprocess.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--train_path', default='data/train.txt', type=str, required=False, help='训练原始数据位置')
    parser.add_argument('--save_path', default='data/train.pkl', type=str, required=False, help='tokenize的训练数据集')
    args = parser.parse_args()

    # 初始化日志对象
    logger = create_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    logger.info(f"preprocessing data,data path:{args.train_path}, save path:{args.save_path}")

    # 读取训练数据集
    with open(args.train_path, 'rb') as f:
        data = f.read().decode("utf-8")

    # 需要区分linux和windows环境下的换行符。一问一答，为一轮
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")  # 分割成一段，一段对话的。
    else:
        train_data = data.split("\n\n")
    logger.info(f"there are {len(train_data)} dialogue in dataset")

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    for index, dialogue in enumerate(tqdm(train_data, total=len(train_data), ascii=' >>', colour='green', dynamic_ncols=True)):
        if "\r\n" in data:
            utterances = dialogue.split("\r\n")
        else:
            utterances = dialogue.split("\n")

        input_ids = [cls_id]  # 每个dialogue以[CLS]开头
        for utterance in utterances:
            input_ids += tokenizer.encode(utterance, add_special_tokens=False)
            input_ids.append(sep_id)  # 每个utterance之后添加[SEP]，表示utterance结束
        dialogue_len.append(len(input_ids))
        dialogue_list.append(input_ids)

    len_mean = np.mean(dialogue_len)  # 统计平均长度
    len_median = np.median(dialogue_len)  # 统计长度的中位数
    len_max = np.max(dialogue_len)  # 统计长度的最大值
    logger.info(f"mean of dialogue len:{len_mean:.3f},median of dialogue len:{len_median},max len:{len_max}")

    # 保存处理好的结果
    with open(args.save_path, "wb") as f:
        pickle.dump(dialogue_list, f)
    logger.info(f"finish preprocessing data,the result is stored in {args.save_path}")


if __name__ == '__main__':
    preprocess()
