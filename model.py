# -*- coding: utf-8 -*-
# @Time : 2022/3/9 17:35
# @Author: Shelly Tang
# @File: model.py
# @Function: bert model

import os
import logging
from tqdm import tqdm, trange
import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from util import write_valid_result

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def accuracy(outputs, labels):
    return np.sum(outputs == labels)


class BertModel(object):
    def __init__(self, args, device, n_gpu, num_labels):
        self.args = args
        self.device = device
        self.n_gpu = n_gpu
        self.num_labels = num_labels
        self.criterion = torch.nn.BCEWithLogitsLoss()       # 二分类交叉熵，含sigmoid版本（用于multi-label）

        self.model = BertForSequenceClassification.from_pretrained(args.bert_model,
                    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(self.args.local_rank),
                    num_labels=self.num_labels)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)

        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            self.model = DDP(self.model)
        elif n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def train_model(self, train_features, eval_features, fold_index):

        num_train_steps = int(len(train_features) / self.args.train_batch_size * self.args.num_train_epochs)

        # 通用的optimizer写法：bias和 LayerNorm没有权重衰减
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        t_total = num_train_steps
        if self.args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)

        global_step = 0

        logger.info("***** Running training of batch %d *************************" % fold_index )
        logger.info("  Train Num examples = %d", len(train_features))
        logger.info("  Train Batch size = %d", self.args.train_batch_size)
        logger.info("  Train Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if self.args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=self.args.train_batch_size)  # 采样成batch

        self.model.train()

        tr_loss_arr = []
        eval_loss_arr = []
        eval_acc_arr = []
        eval_f1_macro_arr = []
        eval_f1_micro_arr = []
        for epoch_index in trange(int(self.args.num_train_epochs), desc="Train Epoch"):
            logger.info("***** Running training of batch %d, epoch %d *************************" % (fold_index, epoch_index))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # for step, batch in enumerate(train_dataloader):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                outputs = self.model(input_ids, segment_ids, input_mask)
                loss = self.criterion(outputs, label_ids.float())
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # modify learning rate with special warm up BERT uses
                lr_this_step = self.args.learning_rate * warmup_linear(global_step / t_total, self.args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            tr_loss_arr.append(round(tr_loss/nb_tr_steps, 4))
            logger.info("Batch %d, Epoch %d: train loss is %.4f" % (fold_index, epoch_index, tr_loss/nb_tr_steps))

            # 每跑完一个epoch就eval一次
            eval_loss, eval_accuracy, eval_f1_macro, eval_f1_micro, \
            outputs, targets = self.eval_model(eval_features, fold_index=fold_index+1)
            logger.info("Batch %d, Epoch %d: eval loss is %.4f, eval acc is %.4f" % (fold_index, epoch_index, eval_loss, eval_accuracy))
            logger.info("eval f1_macro is %.4f, eval f1_micro is %.4f" %(eval_f1_macro, eval_f1_micro))

            eval_loss_arr.append(eval_loss)
            eval_acc_arr.append(eval_accuracy)
            eval_f1_macro_arr.append(eval_f1_macro)
            eval_f1_micro_arr.append(eval_f1_micro)

            # Shelly Tang debug
            if epoch_index == int(self.args.num_train_epochs)-1:
                # logger.info(len(outputs), len(outputs[0]), len(targets), len(targets[0]))
                # outputs1 = np.array(outputs).reshape(len(outputs)).tolist()
                # targets1 = np.array(targets).reshape(len(targets)).tolist()
                # confusion_matrix1 = self.get_confusion_matrix(outputs1, targets1, labels=[0, 1, 2, 3, 4, 5, 6]).tolist()
                # logger.info(confusion_matrix1)
                write_valid_result(self.args, outputs)

        return tr_loss_arr, global_step, eval_loss_arr, eval_acc_arr, eval_f1_macro_arr, eval_f1_micro_arr

    def eval_model(self, eval_features, fold_index):
        logger.info("***** Running evaluation %d *************************" % fold_index)
        logger.info("  Eval Num examples = %d", len(eval_features))
        logger.info("  Eval Batch size = %d", self.args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        self.model.eval()
        eval_loss, eval_accuracy, eval_f1_micro, eval_f1_macro = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        outputs_arr = []
        label_arr = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Eval Iteration"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                # tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)
            tmp_eval_loss = self.criterion(logits, label_ids.float())

            # logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            # outputs = np.argmax(logits, axis=1)                 # Shelly Tang multi label to do
            # tmp_eval_accuracy = accuracy(outputs, label_ids)    # Shelly Tang multi label to do

            outputs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            outputs = (np.array(outputs) > 0.6).astype(int)       # Shelly Tang
            accuracy, micro_f1, macro_f1 = self.get_metrics(outputs.tolist(), label_ids.tolist())


            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += accuracy
            eval_f1_macro += macro_f1
            eval_f1_micro += micro_f1

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            # Shelly Tang debug
            outputs_arr.extend(outputs.tolist())
            label_arr.extend(label_ids.tolist())

        eval_loss = round(eval_loss / nb_eval_steps, 4)
        eval_accuracy = round(eval_accuracy / nb_eval_steps, 4)
        eval_f1_micro = round(eval_f1_micro / nb_eval_steps, 4)
        eval_f1_macro = round(eval_f1_macro / nb_eval_steps, 4)
        return eval_loss, eval_accuracy, eval_f1_macro, eval_f1_micro, outputs_arr, label_arr

    def predict_model(self, test_features, fold_index, label_list):
        id2label = {i: label for i, label in enumerate(label_list)}

        logger.info("***** Running prediction %d *************************" % fold_index)
        logger.info("  Test Num examples = %d", len(test_features))
        logger.info("  Test Batch size = %d", self.args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.eval_batch_size)

        self.model.eval()
        test_outputs = []
        for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            # single-label
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1).tolist()  # Shelly Tang multi label to do
            output_temp_arr = []
            for output in outputs:
                output_temp_arr.append([id2label[output]])

            # multi_label
            # outputs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            # outputs = (np.array(outputs) > 0.6).astype(int)
            # output_temp_arr = []
            # for output in outputs:
            #     output = np.array(output)
            #     output = np.where(output == 1)[0].tolist()
            #     output_temp = [id2label[i] for i in output]
            #     output_temp_arr.append(output_temp)

            test_outputs.extend(output_temp_arr)
        return test_outputs


    def save_model(self, index):
        """
        Save a trained model
        :param index: No. index model
        :return: none
        """
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        temp_path = "pytorch_model_" + str(index) + ".bin"
        output_model_file = os.path.join(self.args.output_dir, temp_path)
        torch.save(model_to_save.state_dict(), output_model_file)

    def load_model(self, output_model_file):
        # Load a trained model that you have fine-tuned
        output_model_file = os.path.join(self.args.output_dir, output_model_file)
        model_state_dict = torch.load(output_model_file)
        self.model = BertForSequenceClassification.from_pretrained(self.args.bert_model, state_dict=model_state_dict,
                                                                   num_labels=self.num_labels)
        self.model.to(self.device)

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_confusion_matrix(self, outputs, targets, labels):
        confusion_matrix1 = confusion_matrix(targets, outputs)
        # report = classification_report(targets, outputs, target_names=labels)
        return confusion_matrix1