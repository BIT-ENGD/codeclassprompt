
from yacs.config import CfgNode
from openprompt.data_utils import FewShotSampler
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputExample
from openprompt.pipeline_base import PromptDataLoader, PromptModel, PromptForClassification
from typing import *
import torch
# from openprompt.utils.custom_tqdm import tqdm
from tqdm import tqdm



def calibrate(prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:
    r"""Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`_
    
    Args:
        prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
        dataloader (:obj:`List`): the dataloader to conduct the calibrate, could be a virtual one, i.e. contain an only-template example.
    
    Return:
        (:obj:`torch.Tensor`) A tensor of shape  (vocabsize) or (mask_num, vocabsize), the logits calculated for each word in the vocabulary
    """
    all_logits = []
    prompt_model.eval() #批大小36,200个样本
    for batch in tqdm(dataloader,desc='ContextCali'): 
        batch = batch.to(prompt_model.device)
        logits = prompt_model.forward_without_verbalize(batch) # batch_size,vocabulary_size
        all_logits.append(logits.detach()) # 把所有样本所mask对应的所有词的概率存起来， 最后是 total_number,vocabulary_size
    all_logits = torch.cat(all_logits, dim=0)# concat 拼一起
    return all_logits

