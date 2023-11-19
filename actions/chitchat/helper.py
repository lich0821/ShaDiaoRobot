#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from .seq2seq import Seq2Seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="serve", help="Train or Serve")
    parser.add_argument("--data_path", "-p", type=str, default="dataset/", help="dataset path")
    parser.add_argument("--log_dir", type=str, default="logs/", help="dataset path")
    parser.add_argument("--model_data", type=str, default="model_data/", help="mode output path")
    parser.add_argument("--dataset", "-n", type=str, default="xiaohuangji50w", help="Train or Serve")
    parser.add_argument("--e", type=str, default="E", help="start flag of conversation")
    parser.add_argument("--m", type=str, default="M", help="start flag of conversation")
    parser.add_argument("--vacab_size_in", "-i", type=int, default=20000, help="vacabulary input size")
    parser.add_argument("--vacab_size_out", "-o", type=int, default=20000, help="vacabulary output size")
    parser.add_argument("--layer_size", type=int, default=256, help="layer size")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--layers", type=int, default=2, help="layers")
    parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dimention")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--max_length", type=int, default=32, help="max length of input")

    args, _ = parser.parse_known_args()
    config = vars(args)

    seq2seq = Seq2Seq(config)

    if args.mode.lower() == "train":
        seq2seq.train()
    else:
        while True:
            msg = input(">>> ")
            rsp = seq2seq.predict(msg)
            print(rsp)

