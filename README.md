# Chinese_text_classification_pytorch_bert
Chinese text classification (include single-label and multi-label version), using pytorch &amp; BERT

中文文本分类任务（含单标签和多标签分类）。

PyTorch实现，BERT框架，CPU/多卡GPU版本。



## Readme

##### 实验环境

- `pytorch` 1.4.0版本、`python` 3.6版本、`pytorch_pretrained_bert` 0.6.2版本。【严格匹配！】
- `argparse`库、`pandas`库、`glob`库、`sklearn`库、`numpy`库（详见`requirements.txt`）。

##### 实验工具

Pycharm

##### 文件组织

- `data`文件夹：
  - `Hotel_comment`文件夹：酒店评论，二分类；
  - `cnews`文件夹：新闻文本，多分类；
- `mytask_classifier.py`：入口文件
- `config.py`：包含运行时所需参数的定义，参数可通过`run.sh`脚本文件赋值
- `data.py`：包含对原数据集的处理，形成结构化数据集
- `model.py`：包含使用BERT实现文本分类的模型代码，含有单标签/多标签两种实现；
- `preprocess.py`：包含BERT的输入预处理
- `util.py`：包含有用函数
- `run.sh`：脚本文件，可在Linux下运行，包含参数的赋值




##### Environments

- `pytorch` == 1.4.0, `python` == 3.6, `pytorch_pretrained_bert` == 0.6.2 (Version needs to match exactly!)
- `argparse`, `pandas`, `glob`, `sklearn`, `numpy`（Please refer to `requirements.txt`）.

##### IDE

Pycharm

##### File structure

- `data` folder: 
  - `Hotel_comment` folder: Chinese comment of a hotel，binary classification task;
  - `cnews` folder: Chinese news paragraph，multi classification task;
- `mytask_classifier.py`: entrance file
- `config.`: includes the definitions of parameters required at runtime, parameters can be assigned through the `run.sh` script file
- `data.py`: includes the processing of the original data set to form a structured data set
- `model.py`：includes the model code for text classification with BERT, which contains the single-label/multi-label implementations;
- `preprocess.py`：includes the input preprocessing for BERT
- `util.py`：includes some useful functions
- `run.sh`：script file, runnable under Linux, containing parameter assignments
