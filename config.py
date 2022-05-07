# -*- coding: utf-8 -*-
# @Time : 2022/3/7 15:07
# @Author: Shelly Tang
# @File: config.py
# @Function: bert configuration

import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        ## Required parameters
        parser.add_argument("--data_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        # parser.add_argument("--label_dir",
        #                     default=None,
        #                     type=str,
        #                     required=True,
        #                     help="The label of dataset. Should contain the .csv files for the task.")
        parser.add_argument("--log_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The path of the log to save. Should contain the .log files for the task.")
        parser.add_argument("--bert_model", default=None, type=str, required=True,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default=None,
                            type=str,
                            required=True,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default=None,
                            type=str,
                            required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")

        # Shelly Tang add 'k' for k_fold in data.py
        # parser.add_argument("--k",
        #                     default=5,
        #                     type=int,
        #                     required=True,
        #                     help="The k value for k_fold used in data.py."
        #                     )

        # Shelly Tang add 'part_index' for train_part_num in data.py
        parser.add_argument("--part_index",
                            default=5,
                            type=int,
                            required=True,
                            help="The part_index value for train used in data.py."
                            )


        ## Other parameters
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        parser.add_argument("--do_train",
                            default=False,
                            help="Whether to run training.")
        parser.add_argument("--do_eval",
                            default=False,
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=3.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--no_cuda",
                            default=False,
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus(set running on specific gpu)")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

