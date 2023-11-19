# -*- coding: utf-8 -*-

import os

from .seq2seq import Seq2Seq

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

config = {"mode": "serve",
          "data_path": f"{ROOT_PATH}/dataset/",
          "log_dir": f"{ROOT_PATH}/logs/",
          "model_data": f"{ROOT_PATH}/model_data",
          "dataset": "xiaohuangji50w",
          "e": "E",
          "m": "M",
          "vacab_size_in": 20000,
          "vacab_size_out": 20000,
          "layer_size": 256,
          "batch_size": 128,
          "embedding_dim": 64,
          "epochs": 10,
          "max_length": 20}
