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

将多层数据进行打包，目前只有最后一层有效果（实际上没打包）
'''
    
class BundleAttention(nn.Module):
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
        self.fc = nn.Linear(hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        
        self.id2rel = {}
        self.drop = nn.Dropout()
        if use_diag:
            self.use_diag = True
            self.diag = nn.Parameter(torch.ones(hidden_size))
        else:
            self.use_diag = False

    
    def forward(self, label, rep, train=True, bag_size=0):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        Dirty hack:
            When the encoder is BERT, the input is actually token, att_mask, pos1, pos2, but
            since the arguments are then fed into BERT encoder with the original order,
            the encoder can actually work out correclty.
        """
        # Attention
        if train:
           
            batch_size = rep.size(0)
            query = label.unsqueeze(1) # (B, 1)
            att_mat = self.fc.weight[query] # (B, 1, H)
            if self.use_diag:
                att_mat = att_mat * self.diag.unsqueeze(0)
            rep = rep.view(batch_size, bag_size, -1)
            att_score = (rep * att_mat).sum(-1) # (B, bag)
            softmax_att_score = self.softmax(att_score) # (B, bag)
            bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1) # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:

               
                batch_size = rep.size(0)   # total/ bag_size batch_size=total
                bag_size =rep.size(1)
                att_mat = self.fc.weight.transpose(0, 1)
                if self.use_diag:
                    att_mat = att_mat * self.diag.unsqueeze(1) 
                att_score = torch.matmul(rep, att_mat) # (nsum, H) * (H, N) -> (nsum, N)
                att_score = att_score.view(batch_size, bag_size, -1) # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) # (B, N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep) # (B, N, bag) * (B, bag, H) -> (B, N, H)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) # (B, (each rel)N)
        return bag_logits

class CodePromptBundleAttention(nn.Module):
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
        self.attention = BundleAttention(self.hidden_size,self.verbalizer.num_classes,use_diag=True)
        self.args = args
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
        # outputs = outputs.hidden_states[len(outputs.hidden_states)-1]
        # outputs_at_mask = self.extract_at_mask(outputs, batch) # batch_size * hidden_size


        if self.args.layer_num == -1:
            start_layer = len(outputs.hidden_states)-1
        else:
            start_layer= len(outputs.hidden_states)-self.args.layer_num
        

        outputs_at_mask=[]
        for layer in outputs.hidden_states[start_layer:]:
            outputs_at_mask.append(self.extract_at_mask(layer, batch)) # batch_size * hidden_size
        logits=torch.stack(outputs_at_mask,dim=0)
        outputs_at_mask= logits.transpose(0,1)
        # if train :

        #     #outputs_at_mask=outputs_at_mask.view(-1,self.args.shot,self.hidden_size)
        #     indexs=[]
        #     for index in range(0,batch["label"].size(0), self.args.shot):
        #         indexs.append(index)

        #     label=torch.index_select(batch["label"],0,torch.tensor(indexs).to(self.args.device))
 
        # else:
        #     label=None

        bag_size=len( outputs.hidden_states)-start_layer
        logits=self.attention(batch["label"],outputs_at_mask,train,bag_size)
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


