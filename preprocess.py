# -*- coding: utf-8 -*-
# @Time : 2022/3/7 15:38
# @Author: Shelly Tang
# @File: preprocess.py
# @Function: preprocess data to input

import csv
import os

SCALE = 10


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:        # Shelly Tang
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


# Shelly Tang 2022/2/27 my dataset
class MyTaskProcessor(object):
    """Processor for my own task"""

    def get_train_examples(self, train_set):
        """See base class."""
        return self._create_examples(train_set, "train")


    def get_dev_examples(self, dev_set):
        """See base class."""
        return self._create_examples(dev_set, "dev")


    # def get_test_examples(self, data_dir):
    #     """See base class."""
    #     return self._create_examples(
    #         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """
        Creates examples for the training and dev sets.
        @lines = [filename, label, user_content, cus_content](without title)
        @set_type = "train" / "dev"
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-file%s" % (set_type, i, lines[0])      # Shelly Tang unique id可以改成"train-index-filex"
            if set_type == "test":
                text_a = line[1]
                text_b = line[2]
                label = "0"
            else:
                text_a = line[2]
                text_b = line[3]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# Shelly Tang 2022/3/12 single-label hotel comment, for active learning
class HotelCommentProcessor(DataProcessor):
    """Processor for hotel comment task"""

    def get_train_examples(self, data_dir, batch_index):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train", batch_index)

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eval.csv")), "dev")

    def get_test_examples(self, data_dir, batch_index):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "test", batch_index)

    @staticmethod
    def get_labels():
        return ["0", "1"]

    def _create_examples(self, lines, set_type, batch_index=0):
        """
        Creates examples for the training and dev sets.
        @lines = [id, content, label](without title), id is unique for each data
        @set_type = "train" / "dev" / "test"
        """
        examples = []
        batch_size = int(len(lines) / SCALE)
        if set_type == "train":
            lines = lines[:batch_size*batch_index]
        elif set_type == "test":
            lines = lines[batch_size*(batch_index-1): batch_size*batch_index]
        for (i, line) in enumerate(lines):
            guid = "%s-%s-data%s" % (set_type, i, line[0])      # Shelly Tang unique id可以改成"train-index-id"
            text_a = line[1]
            text_b = None
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# Shelly Tang 2022/3/12 multi-label pulan, for active learning
class PulanProcessor(DataProcessor):
    """Processor for hotel comment task"""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eval.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    @staticmethod
    def get_labels(filename):
        with open(filename, 'r') as f:
            labels = f.read().strip().split('\n')
        return labels

    def _create_examples(self, lines, set_type):
        """
        Creates examples for the training and dev sets.
        @lines = [id, content, label](without title), id is unique for each data
        @set_type = "train" / "dev" / "test"
        """
        examples = []
        labels_list = lines[0][3:]      # label名字
        print(labels_list)
        lines = lines[1:]
        for (i, line) in enumerate(lines):
            guid = "%s-%s-data%s" % (set_type, i, line[0])      # Shelly Tang unique id可以改成"train-index-id"
            text_a = line[1]
            text_b = line[2]
            label = []              # Shelly Tang multi-label todo 2022/3/18
            if set_type!="test":
                for (index, temp_label) in enumerate(line[3: 3+len(labels_list)]):
                    if ".0" in temp_label:
                        temp_label = str(int(float(temp_label)))
                    label.append(labels_list[index]+":"+temp_label)
            else:
                label = [labels_list[0]+":1"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# Shelly Tang 2022/3/27 single-label news, for active learning
class NewsProcessor(DataProcessor):
    """Processor for hotel comment task"""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eval.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    @staticmethod
    def get_labels(filename):
        with open(filename, 'r') as f:
            labels = f.read().strip().split('\n')
        return labels

    def _create_examples(self, lines, set_type):
        """
        Creates examples for the training and dev sets.
        @lines = [id, content, label](without title), id is unique for each data
        @set_type = "train" / "dev" / "test"
        """
        examples = []
        labels_list = lines[0][2:]      # label名字
        print(labels_list)
        lines = lines[1:]
        for (i, line) in enumerate(lines):
            guid = "%s-%s-data%s" % (set_type, i, line[0])      # Shelly Tang unique id可以改成"train-index-id"
            text_a = line[1]
            text_b = None
            label = []              # Shelly Tang multi-label todo 2022/3/18
            if set_type!="test":
                for (index, temp_label) in enumerate(line[3: 3+len(labels_list)]):
                    label.append(labels_list[index]+":"+temp_label)
            else:
                label = [labels_list[0]+":体育"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# 将token_a和token_b进行长度截取，且截取长的
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # Shelly Tang 待优化 todo

    label2id = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)      # 3 for [CLS][SEP][SEP]
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Shelly Tang multi-label todo: 转为one-hot 2022/3/18
        label_ids = [0]*len(label_list)
        for label in example.label:
            if label not in label2id:
                print("Shelly Tang: error - wrong label: %d" %ex_index)
            label_ids[label2id[label]] = 1


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))

    return features