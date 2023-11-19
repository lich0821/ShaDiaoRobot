# -*- coding: utf-8 -*-

import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Disable Tensorflow debug message

import jieba
import tensorflow as tf
from tqdm import tqdm

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)

from .data_processing import Data, add_flag


class Encoder(tf.keras.Model):
    # 定义Encoder类
    # 初始化函数，对默认参数进行初始化
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # 定义调用函数，实现逻辑计算
    def call(self, x, hidden):
        x_emb = self.embedding(x)
        output, state = self.gru(x_emb, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    # 定义bahdanauAttention类，bahdanauAttention是常用的attention实现方法之一
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # 注意力网络的初始化
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 将query增加一个维度，以便可以与values进行线性相加
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # 将quales与hidden_with_time_axis进行线性相加后，使用tanh进行非线性变换，最后输出一维的score
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        # 使用softmax将score进行概率化转换，转为为概率空间
        attention_weights = tf.nn.softmax(score, axis=1)
        # 将权重与values（encoder_out)进行相乘，得到context_vector
        context_vector = attention_weights * values
        # 将乘机后的context_vector按行相加，进行压缩得到最终的context_vector
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        # 初始化batch_sz、dec_units、embedding 、gru 、fc、attention
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, y, hidden, enc_output):
        # 首先对enc_output、以及decoder的hidden计算attention，输出上下文语境向量
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # 对decoder的输入进行embedding
        y = self.embedding(y)
        # 拼接上下文语境与decoder的输入embedding，并送入gru中
        y = tf.concat([tf.expand_dims(context_vector, 1), y], axis=-1)
        output, state = self.gru(y)
        # 将gru的输出进行维度转换，送入全连接神经网络 得到最后的结果
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.fc(output)
        return y, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))


class Seq2Seq(object):
    def __init__(self, config) -> None:
        self.config = config
        vacab_size_in = config['vacab_size_in']
        vacab_size_out = config['vacab_size_out']
        embedding_dim = config['embedding_dim']
        self.units = config['layer_size']
        self.batch_size = config['batch_size']
        self.encoder = Encoder(vacab_size_in, embedding_dim, self.units, self.batch_size)
        self.decoder = Decoder(vacab_size_out, embedding_dim, self.units, self.batch_size)
        self.optimizer = tf.keras.optimizers.Adam()
        # self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.ckpt_dir = self.config["model_data"]
        logging.basicConfig(level=logging.INFO)
        self.LOG = logging.getLogger("Seq2Seq")
        if tf.io.gfile.listdir(self.ckpt_dir):
            self.LOG.info("正在加载模型...")
            self.checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_dir))

        data = Data(config)
        self.dataset, self.tokenizer_in, self.tokenizer_out = data.load()
        self.steps_per_epoch = data.steps_per_epoch

    def loss_function(self, real, pred):
        # 定义损失函数
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # mask掉start,去除start对于loss的干扰
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)  # 将bool型转换成数值
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def training_step(self, inp, targ, targ_lang, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['bos']] * self.batch_size, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)
        step_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return step_loss

    def train(self):
        # 定义训练函数
        # 从训练语料中读取数据并使用预生成词典word2number的转换
        enc_hidden = self.encoder.initialize_hidden_state()
        writer = tf.summary.create_file_writer(self.config["log_dir"])

        self.LOG.info(f"数据目录: {self.config['data_path']}")
        self.LOG.info(f"每个 epoch 训练步数: {self.steps_per_epoch}")

        epoch = 0
        train_epoch = self.config["epochs"]
        while epoch < train_epoch:
            total_loss = 0
            # 进行一个epoch的训练，训练的步数为steps_per_epoch
            iter_data = tqdm(self.dataset.take(self.steps_per_epoch))
            for batch, (inp, targ) in enumerate(iter_data):
                batch_loss = self.training_step(inp, targ, self.tokenizer_out, enc_hidden)
                total_loss += batch_loss
                iter_data.set_postfix_str(f"batch_loss: {batch_loss:.4f}")

            step_loss = total_loss / self.steps_per_epoch
            self.LOG.info(f"Epoch: {epoch+1}/{train_epoch} Loss: {total_loss:.4f} 平均每步 loss {step_loss:.4f}")

            # 将本epoch训练的模型进行保存，更新模型文件
            self.checkpoint.save(file_prefix=os.path.join(self.ckpt_dir, "ckpt"))
            sys.stdout.flush()
            epoch = epoch + 1
            with writer.as_default():
                tf.summary.scalar("loss", step_loss, step=epoch)

    def predict(self, sentence):
        # 定义预测函数，用于根据上文预测下文对话

        # 对输入的语句进行处理，加上start end标示
        max_length = self.config["max_length"]
        sentence = " ".join(jieba.cut(sentence))
        sentence = add_flag(sentence)

        # 进行word2number的转换
        inputs = self.tokenizer_in.texts_to_sequences([sentence])
        inputs = [[x for x in inputs[0] if x if not None]]  # 去掉空值。TODO: 为啥会有空值？

        # 进行padding的补全
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length, padding="post")
        inputs = tf.convert_to_tensor(inputs)

        result = ""
        hidden = [tf.zeros((1, self.units))]  # 初始化一个中间状态

        # 对输入上文进行encoder编码，提取特征
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        # decoder的输入从start的对应Id开始正向输入
        dec_input = tf.expand_dims([self.tokenizer_out.word_index["bos"]], 0)
        # 在最大的语句长度范围内容，使用模型中的decoder进行循环解码
        for _ in range(max_length):
            # 获得解码结果，并使用argmax确定概率最大的id
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            # 判断当前Id是否为语句结束表示，如果是则停止循环解码，否则进行number2word的转换，并进行语句拼接
            if self.tokenizer_out.index_word[predicted_id] == "eos":
                break
            result += str(self.tokenizer_out.index_word[predicted_id])
            # 将预测得到的id作为下一个时刻的decoder的输入
            dec_input = tf.expand_dims([predicted_id], 0)

        return result
