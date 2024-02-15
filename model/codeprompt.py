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
from typing import Union,Dict
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
    

class CodePromptClassification(nn.Module):
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
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer
        hidden_size=1024
        self.w_prompt =nn.Parameter(torch.rand(1,hidden_size)) #config.hidden_size))
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean') #sum
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

    def predict(self):
        pass

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = outputs.hidden_states[len(outputs.hidden_states)-1]
        outputs_at_mask = self.extract_at_mask(outputs, batch) # batch_size * hidden_size
        out=outputs_at_mask*self.w_prompt  #basize_size * hidden_size
        #myclass=torch.argmax(out,1)
        return out
    
        #logits = self.linear(self.dropout(out))
        #return logits
        #return self.loss_fct(out,batch["label"])
   

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template, plm and verbalizer's save methods."""
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template, plm and verbalizer's load methods."""
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)
    #     self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)

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


class CodePromptClassificationEx(nn.Module):
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
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer
        hidden_size=self.prompt_model.plm.config.hidden_size #1024 
       # self.w_prompt =nn.Parameter(torch.rand(1,hidden_size)) #config.hidden_size))
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean') #sum
       
        self.linear = nn.Linear(hidden_size, self.verbalizer.num_classes) # 直接用cls向量接全连接层分类
        self.dropout = nn.Dropout(0.5)
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

    def predict(self):
        pass

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = outputs.hidden_states[len(outputs.hidden_states)-1]
        outputs_at_mask = self.extract_at_mask(outputs, batch) # batch_size * hidden_size
        #out=outputs_at_mask*self.w_prompt  #basize_size * hidden_size
        logits = self.linear(self.dropout(outputs_at_mask))
        return logits
        
   

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    ##  We comment this code since it conflict with [OpenDelta](https://github.com/thunlp/OpenDelta)
    # def state_dict(self, *args, **kwargs):
    #     """ Save the model using template, plm and verbalizer's save methods."""
    #     _state_dict = {}
    #     if not self.prompt_model.freeze_plm:
    #         _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)
    #     _state_dict['template'] = self.template.state_dict(*args, **kwargs)
    #     _state_dict['verbalizer'] = self.verbalizer.state_dict(*args, **kwargs)
    #     return _state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """ Load the model using template, plm and verbalizer's load methods."""
    #     if 'plm' in state_dict and not self.prompt_model.freeze_plm:
    #         self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)
    #     self.template.load_state_dict(state_dict['template'], *args, **kwargs)
    #     self.verbalizer.load_state_dict(state_dict['verbalizer'], *args, **kwargs)

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
class Config(object):
    def __init__(self,hidden_size,dropout,num_class) -> None:
        self.hidden_size=hidden_size
        self.classifier_dropout=dropout
        self.hidden_dropout_prob=0.5
        self.num_labels=num_class
    

class Atten(nn.Module):
    def __init__(self, config):
        super(Atten, self).__init__()
        hidden_size = config.hidden_size #隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels ##类别数

        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size))

    def forward_2(self, x):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        M = self.tanh(x)
        # M = [batch size, 12, hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]
        out = x * alpha
        # out = [batch size, 12, hidden_size]
        out = torch.sum(out, 1)
        # out = [batch size, hidden_size]
        # out = F.gelu(out)
        # out = [batch size, hidden_size]
        out = self.fc(out)
        # out = [batch size, num_classes]
        return out
    def forward(self,x):
        M = self.tanh(x)
        # [batch size, text size, num_directions * hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # [batch size, text size, 1]
        out = x * alpha
        # [batch size, text size, num_directions * hidden_size]
        out = torch.sum(out, 1)
        # [batch size, num_directions * hidden_size]
        out = F.relu(out)
        # [batch size, num_directions * hidden_size]
        out = self.fc(out)
        # [batch size, num_classes]
        return out

class CodePromptAttentionPrevious(nn.Module):
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
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer
        hidden_size=self.prompt_model.plm.config.hidden_size #1024 
    
       # self.w_prompt =nn.Parameter(torch.rand(1,hidden_size)) #config.hidden_size))
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean') #sum
       
        self.linear = nn.Linear(hidden_size, self.verbalizer.num_classes) # 直接用cls向量接全连接层分类
        self.dropout = nn.Dropout(0.5)
        config=Config(hidden_size,0.5,self.verbalizer.num_classes)
        self.attention = Atten(config)
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

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = outputs.hidden_states[len(outputs.hidden_states)-1]
        outputs_at_mask = self.extract_at_mask(outputs, batch) # batch_size * hidden_size
              
        outputs_at_mask=outputs_at_mask.unsqueeze(0)
        logits=self.attention(outputs_at_mask)
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
