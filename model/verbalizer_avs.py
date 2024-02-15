from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.data_utils import InputFeatures
from openprompt import Verbalizer
from typing import List, Optional, Dict
import random
from util.mylogger import *
import numpy as np
from collections import Counter
class VerbalOptimizerEx(object):
    
    def __init__(self, word2idx: Dict[str, int], labels: List[str], logits_list: List[np.ndarray],
                 expected: Dict[str, np.ndarray]):
        self.word2idx = word2idx
        self.labels = labels
        self.expected = expected

        #softmax 处理
        logits_list =  np.exp(logits_list) # [np.exp(logits) for logits in logits_list]
        self.probs = logits_list / np.expand_dims(np.sum(logits_list, axis=1), axis=1) #original probs_list
        

    def _get_candidates(self, num_candidates: int) -> Dict[str, List[str]]:
        if num_candidates <= 0:
            return {label: self.word2idx.keys() for label in self.labels}

        scores = {label: Counter() for label in self.labels}
        probs=self.probs
        for label in self.labels:
            for word, idx in self.word2idx.items():
                test1=probs[:, idx]
                test=np.log(probs[:, idx]) * self.expected[label] #取出每个样本中的某个word_id,即idx对应的概率 ，然后乘分类表（只要本类别的），再求个和，得到本idx的得分 .按位置相乘，hadmar积
                score = np.sum(np.log(probs[:, idx]) * self.expected[label])# log-softmax ， 似然估计
                scores[label][word] += score

        return {label: [w for w, _ in scores[label].most_common(num_candidates)] for label in self.labels}

    def _get_top_words(self, candidates: Dict[str, List[str]], normalize: bool = True, words_per_label: int = 10,
                       score_fct: str = 'llr') -> Dict[str, List[str]]:

        scores = {label: Counter() for label in self.labels}
        probs=self.probs
        for label in self.labels:
        
            for word in candidates[label]: #对分类中的每个侯选词进行处理
                idx = self.word2idx[word] #得到侯选词的id
                if score_fct == 'llr':
                    scores[label][word] += self.log_likelihood_ratio(probs[:, idx], self.expected[label], normalize)
                elif score_fct == 'ce':
                    scores[label][word] += self.cross_entropy(probs[:, idx], self.expected[label], normalize)
                else:
                    raise ValueError(f"Score function '{score_fct}' not implemented")

        return {label: [w for w, _ in scores[label].most_common(words_per_label)] for label in self.labels}

    @staticmethod
    def log_likelihood_ratio(predictions: np.ndarray, expected: np.ndarray, normalize: bool) -> float: # https://blog.csdn.net/redhatforyou/article/details/104052951
        #prediction:所有样本中某个词的预测概率（softmax后的）， expected 本类标记，如果属于本类，则1, 否则 为 0
        scale_factor = sum(1 - expected) / sum(expected) if normalize else 1   #是否归一化，非本类的数量与本类的数量之比。
        pos_score = scale_factor * (np.sum(np.log(predictions) * expected) - np.sum(np.log(1 - predictions) * expected))  #正例得分
        neg_score = np.sum(np.log(1 - predictions) * (1 - expected)) - np.sum(np.log(predictions) * (1 - expected))   #负例 得分
        return pos_score + neg_score
 
    @staticmethod
    def cross_entropy(predictions: np.ndarray, expected: np.ndarray, normalize: bool) -> float:
        scale_factor = sum(1 - expected) / sum(expected) if normalize else 1
        pos_score = scale_factor * np.sum(np.log(predictions) * expected)
        neg_score = np.sum(np.log(1 - predictions) * (1 - expected))
        return pos_score + neg_score

    def find_verbalizer(self, words_per_label: int = 10, num_candidates: int = 1000, normalize: bool = True, score_fct: str = 'llr'):
        if score_fct == 'random':
            return {label: random.sample(self.word2idx.keys(), words_per_label) for label in self.labels}

        candidates = self._get_candidates(num_candidates=num_candidates)
        return self._get_top_words(candidates=candidates, normalize=normalize, words_per_label=words_per_label,
                                   score_fct=score_fct)
