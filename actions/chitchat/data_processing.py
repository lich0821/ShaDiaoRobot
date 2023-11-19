#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import logging
import os

import jieba
import tensorflow as tf
from tqdm import tqdm


def add_flag(w):
    return "<bos> " + w + " <eos>"


class Data(object):
    def __init__(self, config) -> None:
        self.config = config
        self.seq_path = config["data_path"] + config["dataset"] + ".data"
        self.conv_path = config["data_path"] + config["dataset"] + ".conv"
        self.conv_size = os.path.getsize(self.conv_path)
        self.vacab_path_in = config["data_path"] + config["dataset"] + ".vin"
        self.vacab_path_out = config["data_path"] + config["dataset"] + ".vout"
        self.max_length = config["max_length"]
        self.batch_size = config["batch_size"]
        self.LOG = logging.getLogger("Data")
        logging.basicConfig(level=logging.INFO)
        jieba.setLogLevel(logging.INFO)  # Disable debug info

    def create_sequences(self):
        if os.path.exists(self.seq_path):  # Skip if processed data exists
            return

        # 判断训练语料文件是否存在，如果不存在则进行提醒
        if not os.path.exists(self.conv_path):
            self.LOG.info("找不到语料文件，请检查路径")
            exit()

        self.LOG.info("正在处理语料")
        # 打开需要处理的语料，逐条读取并进行数据处理, 新建一个文件，用于存放处理后的对话语料
        with tqdm(total=self.conv_size) as pbar, open(self.conv_path, encoding="utf-8") as fin, open(self.seq_path, "w") as fout:
            one_conv = ""  # 存储一次完整对话
            for line in fin:
                pbar.update(len(line.encode("utf-8")))
                line = line.strip("\n")
                # line = re.sub(r"[%s]+" % punctuation, "", line)  # 去除标点符号
                if not line:
                    continue
                # 判断是否为一段对话的开始，如果是则把刚刚处理的语料保存下来
                if line[0] == self.config["e"]:
                    if one_conv:
                        fout.write(one_conv[:-1] + "\n")
                    one_conv = ""
                # 判断是否正在处理对话语句，如果是则进行语料的拼接处理 以及分词
                elif line[0] == self.config["m"]:
                    one_conv = one_conv + str(" ".join(jieba.cut(line.split(" ")[1]))) + "\t"  # 存储一次问或答

    def _create_vacab(self, lang, vocab_path, vocab_size):
        if os.path.exists(vocab_path):  # Skip if exists
            return

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<UNK>")
        tokenizer.fit_on_texts(lang)
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json(ensure_ascii=False))

        self.LOG.info(f"正在保存: {vocab_path}")

    def create_vacabularies(self):
        if os.path.exists(self.vacab_path_in) and os.path.exists(self.vacab_path_out):  # Skip if exists
            return

        self.LOG.info(f"正在创建字典")
        lines = io.open(self.seq_path, encoding="UTF-8").readlines()
        word_pairs = [[add_flag(w) for w in l.split("\t")] for l in lines]
        input, target = zip(*word_pairs)
        self._create_vacab(input, self.vacab_path_in, self.config["vacab_size_in"])
        self._create_vacab(target, self.vacab_path_out, self.config["vacab_size_out"])

    def _tokenize(self, path):
        # 定义word2number函数，通过对语料的处理提取词典，并进行word2number处理以及padding补全
        # 从词典中读取预先生成tokenizer的config，构建词典矩阵
        with open(path, "r", encoding="utf-8") as f:
            tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
        # 利用词典进行word2number的转换以及padding处理
        return tokenizer

    def process(self):
        self.create_sequences()
        self.create_vacabularies()

    def load(self):
        self.process()
        lines = io.open(self.seq_path, encoding="UTF-8").readlines()
        word_pairs = [[add_flag(w) for w in l.split("\t")] for l in lines]
        words_in, words_out = zip(*word_pairs)
        tokenizer_in = self._tokenize(self.vacab_path_in)
        tokenizer_out = self._tokenize(self.vacab_path_out)

        tensor_in = tokenizer_in.texts_to_sequences(words_in)
        tensor_out = tokenizer_out.texts_to_sequences(words_out)

        tensor_in = tf.keras.preprocessing.sequence.pad_sequences(tensor_in, maxlen=self.max_length, padding="post")
        tensor_out = tf.keras.preprocessing.sequence.pad_sequences(tensor_out, maxlen=self.max_length, padding="post")

        self.steps_per_epoch = len(tensor_in) // self.batch_size
        BUFFER_SIZE = len(tensor_in)
        dataset = tf.data.Dataset.from_tensor_slices((tensor_in, tensor_out)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset, tokenizer_in, tokenizer_out
