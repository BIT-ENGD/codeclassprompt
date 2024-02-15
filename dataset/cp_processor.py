import os
import json, csv
from openprompt.utils.logging import logger
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


class SmellProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["good","bad"]

    def get_examples(self, data_dir, split):
        pwd=os.getcwd()
        path = os.path.join(data_dir, "SMELL_{}.csv".format(split))
        path = pwd+path
        examples = []
        all_label=set()
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0 :
                    continue  # skip the title
                code, label = row
                label=int(label)
                all_label.add(label)
                text_a = code #headline.replace('\\', ' ')
                text_b = "" #body.replace('\\', ' ')
                example = InputExample(guid=str(idx-1), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples

class CommentProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["lable0","label1","label2","label3","label4","label5","label6","label7","label8","label9","label10","label11","label12","label13","label14","label15"]

    def get_examples(self, data_dir, split):
        

        pwd=os.getcwd()
        path = os.path.join(data_dir, "COMMENT_{}.csv".format(split))
        path = pwd+path
        examples = []
        all_label=set()
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0 :
                    continue  # skip the title
                code, label = row
                label=int(label)
                all_label.add(label)
                text_a = code #headline.replace('\\', ' ')
                text_b = "" #body.replace('\\', ' ')
                example = InputExample(guid=str(idx-1), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples


class SatdProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["none","tech_debt"]

    def get_examples(self, data_dir, split):
        pwd=os.getcwd()
        path = os.path.join(data_dir, "SATD_{}.csv".format(split))
        path = pwd+path
        examples = []
        all_label=set()
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0 :
                    continue  # skip the title
                code, label = row
                label=int(label)
                all_label.add(label)
                text_a = code #headline.replace('\\', ' ')
                text_b = "" #body.replace('\\', ' ')
                example = InputExample(guid=str(idx-1), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples


class CodeProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = ["label0", "label1", "label2", "label3", "label4", "label5", "label6", "label7", "label8", "label9", "label10", "label11", "label12", "label13", "label14", "label15", "SQL", "label17", "label18",]

    def get_examples(self, data_dir, split):
        pwd=os.getcwd()
        path = os.path.join(data_dir, "CODE_{}.csv".format(split))
        path = pwd+path
        examples = []
        all_label=set()
        max_len=0
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0 :
                    continue  # skip the title
                code, label = row
                label=int(label)
                if len(code)> max_len:
                    max_len=len(code)
                all_label.add(label)
                text_a = code #headline.replace('\\', ' ')
                text_b = "" #body.replace('\\', ' ')
                example = InputExample(guid=str(idx-1), text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
        return examples



PROCESSORS = {
    "CODE": CodeProcessor,
    "SMELL": SmellProcessor,
    "COMMENT": CommentProcessor,
    "SATD": SatdProcessor,

}
