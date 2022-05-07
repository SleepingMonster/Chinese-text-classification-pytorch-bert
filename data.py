# -*- coding: utf-8 -*-
# @Time : 2022/3/5 12:14
# @Author: Shelly Tang
# @File: data.py
# @Function: data.py, deal with the dataset

import glob
import pandas as pd
import random
import csv
import os
import numpy as np
from sklearn.metrics import confusion_matrix

KeyContentLen = 512
SplitDataN = 2
PART_NUM = 10

class Data(object):
    def __init__(self, file_path, label_path):
        # org数据集包括空数据
        self.org_dataset = self.getContent(file_path)
        self.org_label = self.getLabel(label_path)
        # 剔除空数据后：
        self.dataset = self.processData2()
        # shuffle, splitdata
        self.shuffleData()      # shuffle
        self.writeDataset()     # output to csv

    @staticmethod
    def getContent(file_path):
        """
        获得原始对话数据
        :param file_path: 文件夹位置（文件已改名）
        :return: dataset[N, data], data[n_line, content]
        """
        dataset = []
        file_number = len(glob.glob(pathname=file_path+'/*.txt'))  # 获取filepath文件夹下的文件个数
        for i in range(file_number):
            file_name = file_path+"/"+str(i+1)+".txt"
            with open(file_name, "r", encoding='utf-8') as f:
                data = []
                for line in f.readlines():
                    line = line.strip('\n')
                    data.append(line)
                dataset.append(data)
        return dataset

    @staticmethod
    def getLabel(label_path):
        label = pd.read_csv(label_path, usecols=['label'])      # 按列读取csv
        return list(label['label'])     # pd对象转list返回

    @staticmethod
    def getKeyContent(content):
        """
        只是取合并customer和user之后的最后K个字输入
        :param content: 字符串，整个数据的所有内容（连在一起）
        :return: 最后的K个字
        """
        result = ""
        if len(content) > KeyContentLen:
            result = content[-KeyContentLen:]
        else:
            result = content
        return result

    @staticmethod
    def getKeyContent2(content):
        """
        取最后K个字作为关键内容，且分开角色
        :param content: [[content, role="user"/"customer"]]
        :return: @content_user:str, @content_customer:str, 分别是K范围内concat在一起的内容
        """
        temp = 0
        i = 1
        user_content = ""
        cus_content = ""
        while i < len(content)+1 and temp+len(content[-i][0]) <= KeyContentLen:
            if content[-i][1] == "user":
                user_content = content[-i][0] + user_content
            else:
                cus_content = content[-i][0] + cus_content
            i += 1
        return user_content, cus_content

    def processData(self):
        """
        处理数据集，如删掉前面的user, customer和时间，然后获取关键内容（如最后的K个字）
        :return: processed dataset
        """
        dataset = []
        for i in range(len(self.org_dataset)):
            data = self.org_dataset[i]
            new_data_content = ""
            if len(data) == 0:
                continue
            for j in range(len(data)):
                line = data[j]
                content = line.split(',')       # 只取具体内容，忽略角色和时间
                new_data_content = new_data_content + content[2]        # 先将全部连成一个字符串
                if len(content) > 3:        # 报错处理，看content有无英文逗号
                    print("Error: English comma exists in content.")

            key_content = self.getKeyContent(new_data_content)
            new_data = [i+1, new_data_content, str(self.org_label[i]), key_content]
            dataset.append(new_data)
        return dataset

    def processData2(self):
        """
        处理数据集，如删掉前面的user, customer和时间，然后获取关键内容（如最后的K个字），且要将user和customer分开，上限为K
        :return: processed dataset
        """
        dataset = []
        for i in range(len(self.org_dataset)):
            data = self.org_dataset[i]
            new_data = []       # [[content, role="user"/"customer"]]
            if len(data) == 0:
                continue
            for j in range(len(data)):
                line = data[j]
                content = line.split(',')       # 只取具体内容，忽略角色和时间
                if len(content) > 3:        # 报错处理，看content有无英文逗号
                    print("Error: English comma exists in content.")
                if content[0][:4] == "user":
                    new_data.append([content[2],"user"])
                else:
                    new_data.append([content[2], "customer"])

            user_key_content, cus_key_content = self.getKeyContent2(new_data)
            new_data = [i+1, str(self.org_label[i]), user_key_content, cus_key_content]
            dataset.append(new_data)
        return dataset

    def shuffleData(self):
        """
        将数据集[index, content, label]进行shuffle
        :return: shuffle后的数据集
        """
        random.shuffle(self.dataset)

    def writeDataset(self):
        with open("./dataset.csv", "w", newline="") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["fileName", "label", "user_content", "cus_content"])
            csvwriter.writerows(self.dataset)

    def k_fold(self, k, i):
        """
        将dataset分成train set和dev set
        :return: train set, dev set
        """
        total = len(self.dataset)
        step = total // k   # 步长
        start = i*step
        end = start + step
        train_set = self.dataset[:start] + self.dataset[end:]
        dev_set = self.dataset[start:end]
        return train_set, dev_set


class Data2(object):
    # multi-label class data
    def __init__(self, data_dir, full_names, names):
        # 构建原dataset(without label)
        # file_path = os.path.join(data_dir, "text_punc")
        # self.org_dataset = self.getContent(file_path)
        # self.dataset = self.processData()
        # self.writeDataset(self.dataset, full_names)     # output to csv

        self.data_dir = data_dir
        self.full_names = full_names
        self.names = names

    @staticmethod
    def getContent(file_path):
        """
        获得原始对话数据
        :param file_path: 文件夹位置（文件已改名）
        :return: dataset[N, data], data[n_line, content]
        """
        dataset = []
        file_number = len(glob.glob(pathname=file_path + '/*.txt'))  # 获取filepath文件夹下的文件个数
        for i in range(file_number):
            file_name = file_path + "/" + str(i + 1) + ".txt"
            with open(file_name, "r", encoding='utf-8') as f:
                data = []
                for line in f.readlines():
                    line = line.strip('\n')
                    data.append(line)
                dataset.append(data)
        return dataset

    @staticmethod
    def splitDataContent(content):
        """
        将一条数据分割成不同的数据（取偶数次的K个字作为一条新数据）
        例如：本content含有3段KeyContentLen个字对的分段（0,1,2)，则取0,2返回（后序可以调整取得段数）
        :param content: list of [content, role="user"/"customer", total_data_len], total_data_len是到该行的字数累加之和
        :return: @content_user:str, @content_customer:str, 分别是K范围内concat在一起的内容
        """
        total_data_len = content[-1][2]
        k_period_count = total_data_len // KeyContentLen
        k_period_count = k_period_count + 1 if total_data_len%KeyContentLen != 0 else k_period_count
        # print("Shelly Tang: total_data_len is %d, k_period_count is %d" %(total_data_len, k_period_count))

        role_to_index = {"user": 0, "customer": 1}
        result_data_content = []    # list of [起始行号,结束行号,user_content,customer_content]
        cur_content = ["", ""]
        cur_data_start, cur_data_end = 0, KeyContentLen     # 表示当前数据的起始/结束字数的序号
        is_start_flag = False                               # 是否是当前数据所取的第一行
        cur_line_start, cur_line_end = 0, 0                 # 表示当前数据的起始/结束行号的序号

        for line_index in range(len(content)):
            if cur_data_start >= total_data_len:
                # print("Shelly Tang:1")
                break
            line = content[line_index]
            if cur_data_start < line[2] < cur_data_end:     # if处于范围内，将其纳入当前新数据中
                role_index = role_to_index[line[1]]
                cur_content[role_index] += line[0]
                if not is_start_flag:
                    cur_line_start = line_index
                    is_start_flag = True
            elif line[2] >= cur_data_end:                   # 这种情况就是，夸(n, n+k)的后界，但不夸前界
                role_index = role_to_index[line[1]]         # PS第一次大于也纳入当前新数据（后面是会truncate的）
                cur_content[role_index] += line[0]
                cur_line_end = line_index

                result_data_content.append([cur_line_start, cur_line_end, cur_content[0], cur_content[1]])

                is_start_flag = False
                cur_content = ["", ""]
                cur_data_start += KeyContentLen*SplitDataN
                cur_data_end += KeyContentLen*SplitDataN
        if is_start_flag and (len(cur_content[0]) != 0 or len(cur_content[1]) != 0):
            result_data_content.append([cur_line_start, len(content)-1, cur_content[0], cur_content[1]])
        return result_data_content, k_period_count

    def processData(self):
        """
        处理数据集，如删掉前面的user, customer和时间，然后将其中的每K个字作为一条新数据，且要将user和customer分开，上限为K
        :return: processed dataset
        """
        dataset = []
        for file_index in range(len(self.org_dataset)):
        # for file_index in range(1):
            data = self.org_dataset[file_index]
            new_data = []       # [[content, role="user"/"customer"]]
            total_data_len = 0  # 计算文本总长度
            if len(data) == 0:
                continue
            for j in range(len(data)):
                line = data[j]
                content = line.split(',')       # 只取具体内容，忽略角色和时间
                total_data_len += len(content[2])       # 到j文本时的字数累加之和

                if len(content) > 3:        # 报错处理，看content有无英文逗号
                    print("Error: English comma exists in content.")
                if content[0][:4] == "user":
                    new_data.append([content[2], "user", total_data_len])
                else:
                    new_data.append([content[2], "customer", total_data_len])
            # 将当前data分成若干个K个字的片段
            result_data_content, k_period_count = self.splitDataContent(new_data)

            for k_period_index in range(len(result_data_content)):
                cur_k_period = result_data_content[k_period_index]
                cur_data = [file_index+1, k_period_count, 2*k_period_index+1, cur_k_period[0]+1, cur_k_period[1]+1,
                            cur_k_period[2], cur_k_period[3]]
                dataset.append(cur_data)
        return dataset

    def shuffleData(self):
        """
        将数据集[index, content, label]进行shuffle
        :return: shuffle后的数据集
        """
        random.shuffle(self.dataset)

    def writeDataset(self, filepath, dataset, names=None):
        # filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w", newline="", encoding="gbk") as f:
            csvwriter = csv.writer(f)
            if names != None:
                csvwriter.writerow(names)
            csvwriter.writerows(dataset)

    def readDataset(self, filename, names=None):
        filepath = os.path.join(self.data_dir, filename)
        if names == None:
            lines = pd.read_csv(filepath, encoding='gbk')
        else:
            lines = pd.read_csv(filepath, usecols=names, encoding="gbk")
        pd.set_option('display.max_columns', 50)

        lines = lines.values.tolist()
        for (index, line) in enumerate(lines):
            if pd.isnull(lines[index][1]):
                lines[index][1] = ""
            if pd.isnull(lines[index][2]):
                lines[index][2] = ""
        return lines

    def buildDataset(self, filename, start, fold_size=160):
        # filepath = os.path.join(self.data_dir, filename)
        lines = self.readDataset(filename, names=self.names)
        temp_lines = lines[:360]
        random.seed(42)
        random.shuffle(temp_lines)
        for (line_idx, line) in enumerate(temp_lines):
            for (label_idx, label) in enumerate(line[3:]):
                line[label_idx + 3] = str(int(float(label)))

        # # 第一次才要train, valid，且从新的一个标了数据的文件里面划分
        # valid_set = temp_lines[:200]
        # train_set = temp_lines[200:360]
        # valid_path = os.path.join(self.data_dir, "eval.csv")
        # train_path = os.path.join(self.data_dir, "train.csv")
        # self.writeDataset(valid_path, valid_set, self.names)
        # self.writeDataset(train_path, train_set, self.names)

        names = ["file_name", "user_content", "cus_content"]
        lines = self.readDataset("dataset.csv", names=names)
        test_set = lines[start:start + fold_size]
        filepath = os.path.join(self.data_dir, "test.csv")
        self.writeDataset(filepath, test_set, self.names)

    def writeTestResult(self, test_result_fn, test_fn, label_names, names):
        test_df = self.readDataset(test_fn)
        # test_result_df = self.readDataset(test_result_fn)
        test_result_file = os.path.join(self.data_dir, test_result_fn)
        label2id = {label: i+3 for (i, label) in enumerate(label_names)}

        with open(test_result_file, "r", encoding='gbk') as f:
            lines = csv.reader(f)
            # lines = lines
            for (line_index, line) in enumerate(lines):
                for label_pair in line:
                    label = label_pair.split(":")[0]
                    label_index = label2id[label]
                    value = label_pair.split(":")[1]
                    if pd.isnull(test_df[line_index][label_index]):
                        test_df[line_index][label_index] = value
                    else:
                        test_df[line_index][label_index] += "," + value

        self.writeDataset("test_2_new.csv", test_df, names=names)

    def writeTestResult2(self, test_result_arr, label_names, filepath):
        test_df = self.readDataset("test.csv")

        label2id = {label: i+3 for (i, label) in enumerate(label_names)}

        for (line_index, line) in enumerate(test_result_arr):
            for label_pair in line:
                label = label_pair.split(":")[0]
                label_index = label2id[label]
                value = label_pair.split(":")[1]
                if pd.isnull(test_df[line_index][label_index]):
                    test_df[line_index][label_index] = str(value)
                else:
                    test_df[line_index][label_index] += "," + str(value)

        self.writeDataset(filepath, test_df, names=self.names)

    def writeEvalResult(self, test_result_arr, label_names, filepath):
        # test_df = self.readDataset("eval.csv")

        label2id = {label: i for (i, label) in enumerate(label_names)}

        result = []
        for (line_index, line) in enumerate(test_result_arr):
            temp = [""]* len(label_names)
            for label_pair in line:
                label = label_pair.split(":")[0]
                label_index = label2id[label]
                value = label_pair.split(":")[1]
                if len(temp[label_index]) == 0:
                    temp[label_index] = value
                else:
                    temp[label_index] += "," + value
                # test_df[line_index][label_index] = value
            result.append(temp)
        self.writeDataset(filepath, result, names=label_names)

    def get_confusion_matrix(self, outputs, targets, labels):
        confusion_matrix1 = confusion_matrix(targets, outputs)
        # report = classification_report(targets, outputs, target_names=labels)
        return confusion_matrix1

class Data3(object):
    # active learning in standard single-label dataset
    def __init__(self, data_dir, org_data_path):
        self.data_dir = data_dir
        # self.org_data_path = org_data_path

        train_lines = self.readTxt("train.txt", 1400)
        eval_lines = self.readTxt("val.txt", 600)
        self.buildOrgDataset(train_lines, "dataset")
        self.buildOrgDataset(eval_lines, "eval")

        self.buildDataset("dataset.csv", 200, 200)

    def readDataset(self, filename, names=None):
        filepath = os.path.join(self.data_dir, filename)
        if names == None:
            lines = pd.read_csv(filepath, encoding='utf-8')
        else:
            lines = pd.read_csv(filepath, usecols=names, encoding="utf-8")
        pd.set_option('display.max_columns', 50)

        lines = lines.values.tolist()
        for (index, line) in enumerate(lines):
            if pd.isnull(lines[index][1]):
                lines[index][1] = ""
            if pd.isnull(lines[index][2]):
                lines[index][2] = ""
        return lines

    def readTxt(self, filename, line_num):
        filepath = os.path.join(self.data_dir, filename)
        print(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        random.seed(42)
        random.shuffle(lines)
        return lines[:line_num]


    def buildOrgDataset(self, lines, type):
        # train maxlen = 15271, avg len=897
        # eval maxlen = 6164, avg len = 921
        result = []
        max_len = 0
        total_len = 0
        for (index, line) in enumerate(lines):
            line = line.strip().strip('\n')
            content = line.split('\t')[1]
            label = line.split('\t')[0]
            result.append([index, content, label])
            if len(content)>max_len:
                max_len = len(content)
            total_len += len(content)
        print("max len: %d" %(max_len))
        print("avg len: %f" %(total_len/len(result)))
        self.writeDataset(result, os.path.join(self.data_dir, type+".csv"))

    def buildDataset(self, filename, start, fold_size=160):
        # filepath = os.path.join(self.data_dir, filename)
        lines = self.readDataset(filename)

        # # 第一次才要train，且从新的一个标了数据的文件里面划分
        train_set = lines[:fold_size]
        self.writeDataset(train_set, os.path.join(self.data_dir, "train.csv"))

        test_set = lines[start:start + fold_size]
        self.writeDataset(test_set, os.path.join(self.data_dir, "test.csv"))

    def writeDataset(self, dataset, file_name):
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["Index", "Content", "Label"])
            csvwriter.writerows(dataset)

    @staticmethod
    def compare_test(filename):
        diff_count = 0
        index_arr = []
        with open(filename, "r", encoding="utf-8") as f:
            lines = csv.reader(f)
            for (index,line) in enumerate(lines):
                if line[0]!=line[1]:
                    diff_count += 1
                    index_arr.append(index)
        print(diff_count)
        print(index_arr)

    def update_train_dataset(self, test_filename, train_filename, part_index):
        test_file = os.path.join(self.data_dir, test_filename)
        train_file = os.path.join(self.data_dir, train_filename)
        test_result = []
        with open(test_file, "r", encoding="utf-8") as f:
            lines = csv.reader(f)
            lines = list(lines)
            total_count = len(lines)
            print("test total_count=%d" %total_count)
            for (index, line) in enumerate(lines):
                if index < int(0.1*total_count):
                    test_result.append(line[1])
                else:
                    test_result.append(line[0])

        lines = []
        with open(train_file, "r", encoding="utf-8") as f:
            lines = csv.reader(f)
            # for line in lines:
            #     print(line)
            lines = list(lines)
            print(len(lines))
            part_size = int(len(lines) / PART_NUM)
            print(part_size)
            for (index, test_data) in enumerate(test_result):
                lines[part_size*(part_index-1)+index][2] = test_data
        with open(train_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(lines)




if __name__ == '__main__':
    full_names = ["file_name", "period_count", "period_index", "start_line_index","end_line_index","user_content", "cus_content",
                  "is_fluent", "is_normal","stage", "category", "education", "pension", "health", "invest", "deposit"]

    names = ["file_name", "user_content", "cus_content", "is_fluent", "is_normal", "category"]
    label_names = ["is_fluent", "is_normal", "category"]



