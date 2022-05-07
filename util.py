# -*- coding: utf-8 -*-
# @Time : 2022/3/7 17:17
# @Author: Shelly Tang
# @File: util.py
# @Function: some utilized function


import random
import numpy as np
import torch
import time
import os
import logging
import csv
from data import Data2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 新建日志文件
    date_str = time.strftime("%Y-%m-%d", time.localtime())
    log_path = log_path + date_str + "/"
    os.makedirs(log_path, exist_ok=True)

    uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    cur_path = "%s.log" % uuid_str
    log_path += cur_path

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def set_seed(seed=123):
    """
    为CPU和GPU设置随机数种子，使得结果确定（随机结果相同，保证实验可复现）
    :param seed: [int]
    :return: none
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     # CPU
    torch.cuda.manual_seed_all(seed)            # GPU(safe to call without cuda)


def show_valid_result(args, fold_index, result):
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results %d ****************" %fold_index)
        writer.write("Batch %d: \n" % fold_index)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
        writer.write("\n")


def write_test_result(args, test_outputs):
    output_test_file = os.path.join(args.output_dir, "test_result_"+str(args.part_index+1)+".csv")
    # with open(output_test_file, "w", newline="") as f:
    #     csvwriter = csv.writer(f)
    #     csvwriter.writerows(test_outputs)
    full_names = ["file_name", "period_count", "period_index", "start_line_index", "end_line_index", "user_content",
                  "cus_content",
                  "is_fluent", "is_normal", "stage", "category", "education", "pension", "health", "invest", "deposit"]

    # names = ["file_name", "user_content", "cus_content", "stage"]
    # names = ["file_name", "user_content", "cus_content", "education", "pension", "health", "invest", "deposit"]
    # names = ["file_name", "user_content", "cus_content", "is_fluent", "is_normal", "category"]
    names = ["Index", "Content", "Label"]
    data = Data2("./data/Pulan_financial_management", full_names, names)
    # 写入test_file结果
    # label_names = ["stage"]
    # label_names = ["education", "pension", "health", "invest", "deposit"]
    # label_names = ["is_fluent", "is_normal", "category"]
    label_names = ["Label"]
    data.writeTestResult2(test_outputs, label_names, output_test_file)


def write_valid_result(args, test_outputs):
    result = []
    # label_list = ["education:0", "education:1", "pension:0", "pension:1", "health:0", "health:1","invest:0", "invest:1", "deposit:0", "deposit:1"]
    # label_list = ["stage:0", "stage:1", "stage:2", "stage:3", "stage:4", "stage:5", "stage:-1"]
    label_list = ["is_fluent:0", "is_fluent:1", "is_normal:0", "is_normal:1", "category:0", "category:1", "category:-1"]
    label2pair = {index: label for (index, label) in enumerate(label_list)}

    for (line_idx, line) in enumerate(test_outputs):
        temp = []
        for (label_idx, label) in enumerate(line):
            line[label_idx] = str(label)
            if label == 1:
                temp.append(label2pair[label_idx])
        result.append(temp)

    output_test_file = os.path.join(args.output_dir, "valid_result.csv")
    full_names = ["file_name", "period_count", "period_index", "start_line_index", "end_line_index", "user_content",
                  "cus_content",
                  "is_fluent", "is_normal", "stage", "category", "education", "pension", "health", "invest", "deposit"]
    # names = ["file_name", "user_content", "cus_content", "stage"]
    # names = ["file_name", "user_content", "cus_content", "education", "pension", "health", "invest", "deposit"]
    # names = ["file_name", "user_content", "cus_content", "is_fluent", "is_normal", "category"]
    names = ["Index", "Content", "Label"]
    data = Data2("./data/Pulan_financial_management", full_names, names)
    # 写入test_file结果
    # label_names = ["stage"]
    # label_names = ["education", "pension", "health", "invest", "deposit"]
    # label_names = ["is_fluent", "is_normal", "category"]
    label_names = ["Label"]
    data.writeEvalResult(result, label_names, output_test_file)
