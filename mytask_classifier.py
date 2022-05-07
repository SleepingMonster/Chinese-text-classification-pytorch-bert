# -*- coding: utf-8 -*-
# @Time : 2022/3/5 12:15
# @Author: Shelly Tang
# @File: mytask_classifier.py
# @Function: text classifier (my_version, from pytorch_pretrained_bert)。
# Active Learning: no k_fold


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from tqdm import tqdm

import torch

from config import Args
from preprocess import MyTaskProcessor, convert_examples_to_features, HotelCommentProcessor, PulanProcessor, NewsProcessor
import util
from data import Data
from model import BertModel

from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForSequenceClassification
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

epsilon = 1e-8


def initialization(args):
    # 获取processor
    processors = {
        "mytask": MyTaskProcessor,
        "mytask_al_single": NewsProcessor,
        "mytask_al_multilabel": PulanProcessor
    }
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()

    num_labels_task = {
        "mytask": 2,
        "mytask_al_single": 10,
        "mytask_al_multilabel": 7      # Shelly Tang 每次换label都要改
    }
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels(args.data_dir + "/labels.txt")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    # 创建目录output
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 更新args.train和args.eval(argparse 中str转bool)
    args.do_train = True if args.do_train.lower() == 'true' else False
    args.do_eval = True if args.do_eval.lower() == 'true' else False

    return processor, num_labels, label_list, device, n_gpu


def main():
    args = Args().get_parser()      # 创建并获取参数
    util.set_seed(args.seed)        # 设置随机数
    util.set_logger(args.log_dir)   # ./log/

    processor, num_labels, label_list, device, n_gpu = initialization(args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)


    # following is train and eval 训练和验证：
    if args.do_train and args.do_eval:
        # Prepare dataset
        train_examples = processor.get_train_examples(args.data_dir)
        eval_examples = processor.get_dev_examples(args.data_dir)
        test_examples = processor.get_test_examples(args.data_dir)
        # Prepare model
        bert_model = BertModel(args, device, n_gpu, num_labels)

        # # Train starts:
        # Get train features(as the input of bert model)
        logger.info("***** Converting train examples to train features ***********")
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        # Get eval features(as the input of bert model)
        logger.info("***** Converting eval examples to eval features ***********")
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        # Get test features(as the input of bert model)
        logger.info("***** Converting test examples to test features ***********")
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)

        # Train(include eval)
        tr_loss_arr, global_step, eval_loss_arr, eval_acc_arr, eval_f1_macro_arr, eval_f1_micro_arr =\
            bert_model.train_model(train_features, eval_features, fold_index=args.part_index)
        # Save model
        bert_model.save_model(index=args.part_index)

        # Test测试
        test_outputs = bert_model.predict_model(test_features, args.part_index+1, label_list)
        util.write_test_result(args, test_outputs)

        # Show eval result in log and console
        result = {'eval_loss': eval_loss_arr,
                  'eval_accuracy': eval_acc_arr,
                  'eval_f1_macro': eval_f1_macro_arr,
                  'eval_f1_micro': eval_f1_micro_arr,
                  'train_global_step': global_step,
                  'train_loss': tr_loss_arr}

        util.show_valid_result(args, args.part_index, result)
        
    else:       # only test
        test_examples = processor.get_test_examples(args.data_dir)
        bert_model = BertModel(args, device, n_gpu, num_labels)
        bert_model.load_model("pytorch_model_5.bin")

        logger.info("***** Converting test examples to test features ***********")
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        test_outputs = bert_model.predict_model(test_features, args.part_index + 1, label_list)
        util.write_test_result(args, test_outputs)

if __name__ == "__main__":
    main()
