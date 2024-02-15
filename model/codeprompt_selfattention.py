import torch.nn as nn 
import torch 
#torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
from transformers import RobertaPreTrainedModel,RobertaTokenizer,RobertaForMaskedLM,RobertaConfig,RobertaModel
from openprompt import PromptForClassification
from openprompt.prompt_base import Template, Verbalizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.utils import round_list, signature
from openprompt.pipeline_base import PromptModel
from typing import Union,Dict,Any
from openprompt.data_utils import InputExample, InputFeatures
import time
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# model = RobertaModel.from_pretrained("microsoft/codebert-base")
# model.to(device) 

# https://github.com/NTDXYG/Text-Classify-based-pytorch/blob/master/model/TextRNN_Attention.py

# https://github.com/NTDXYG/Text-Classify-based-pytorch/blob/master/model/TextCNN.py

# class  CodePrompt(nn.Module):
#     def __init__(self, plm: PreTrainedModel,
#                  template: Template,
#                  verbalizer:Verbalizer,
#                  freeze_plm: bool = False,
#                  plm_eval_mode: bool=False):
#         super().__init__()
#         self.plm=PromptForClassification(plm,template,verbalizer,freeze_plm,plm_eval_mode)
#         self.verbalizer=verbalizer
#        # self.forward_keys = signature(self.plm.forward).args

#     def forward(self,batch):

#         logits=self.plm.forward_without_verbalize(batch=batch) 

#         return logits
    
'''
简单attention层，将从1到最后一层的数据做sefl-attention

'''
class CPAttention(nn.Module):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, hidden_size, num_class, use_diag=False):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
 
        self.num_class = num_class
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size ))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(hidden_size, self.num_class)

    
    def forward(self,rep):
        # Attention

        M = self.tanh1(rep)
        # [batch size, layer num, hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        if hasattr(self,"args"):
            weigh=alpha[0].squeeze(dim=-1).data.tolist()
            att_file="attention/att_visual_{}_id_{}.txt".format(self.args.dataset,self.args.att_visual_id)
            with open(att_file,"w") as f:
                f.write(self.args.code_txt+"\n")
                weigh=list(map(lambda x:str(x),weigh))
                f.write("["+",".join(weigh)+"]")

            
        
        # [batch size, layer num, 1]
        out = rep * alpha
        # [batch size, layer num,  hidden_size]
        out = torch.sum(out, 1)
        # [batch size,  hidden_size]
        out = F.relu(out)
        # [batch size, hidden_size]
        out = self.fc(out)
        # [batch size, num_classes]
        return out


class CodePromptSelfAttention(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the labels to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 args: Any=None,
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer
        self.hidden_size=self.prompt_model.plm.config.hidden_size #1024 
    
       # self.w_prompt =nn.Parameter(torch.rand(1,hidden_size)) #config.hidden_size))
       # self.loss_fct = nn.CrossEntropyLoss(reduction='mean') #sum
       
       # self.linear = nn.Linear(hidden_size, self.verbalizer.num_classes) # 直接用cls向量接全连接层分类
       # self.dropout = nn.Dropout(0.5)
        self.attention = CPAttention(self.hidden_size,self.verbalizer.num_classes)
        self.args = args
        if args.att_visual:
            self.attention.args=args

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward_with_verbalizer(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the label words (obtained by the current verbalizer).
        """
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits

    def forward(self, batch: Union[Dict, InputFeatures],train=True) -> torch.Tensor:

              
        outputs = self.prompt_model(batch)
        #outputs = outputs.hidden_states[len(outputs.hidden_states)-1]
        outputs_at_mask=[]
        start =self.args.layer_num
        if start <0:
            start = 0
        elif start >= len(outputs.hidden_states):
            start = len(outputs.hidden_states)-1

        for layer in outputs.hidden_states[start:]:
            outputs_at_mask.append(self.extract_at_mask(layer, batch)) # batch_size * hidden_size
        logits=torch.stack(outputs_at_mask,dim=0)
        logits2= logits.transpose(0,1)
        logits=self.attention(logits2)
        return logits
        
   

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer
    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")


