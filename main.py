import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

from util import *
from dataset import *
from model import *
from torch.utils.tensorboard.writer import SummaryWriter
from openprompt.utils.reproduciblity import set_seed
from torch.optim import AdamW 
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.prompts import SoftVerbalizer
from openprompt import PromptForClassification
#from transformers.data.metrics import acc_and_f1, simple_accuracy
from sklearn.metrics import f1_score
from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate
from  tqdm import tqdm
import platform
import os
from importlib import util as imp_util
from transformers import  get_linear_schedule_with_warmup
try:
    imp_util.find_spec('torch_directml')
    found_directml = True
    import torch_directml
except ImportError:
    found_directml = False





#  https://github.com/microsoft/CodeBERT
# 参考： https://huggingface.co/huggingface/CodeBERTa-language-id

def init_optimizer(args,loader,prompt_model):
    
    if args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
        no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1,
            num_warmup_steps=500, num_training_steps=tot_step)
    else:
        optimizer1 = None
        scheduler1 = None


    optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
    if args.optimizer.lower() == "adafactor":
        optimizer2 = Adafactor(optimizer_grouped_parameters2,
                                lr=args.prompt_lr,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
        optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr) # usually lr = 0.5
        scheduler2 = get_linear_schedule_with_warmup(
                        optimizer2,
                        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500



    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(loader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2, 
        num_warmup_steps=0, num_training_steps=tot_step)
    
    return optimizer1,optimizer2,scheduler1,scheduler2

'''
Run a turn for a dataset
'''
def run_train_dataset(args,prompt_model,dsinfo,mytemplate,data_loader):

    forward_keys=["input_ids","inputs_embeds","attention_mask"]

    loss_func = torch.nn.CrossEntropyLoss()


    #wrapped_example = mytemplate.wrap_one_example(dsinfo['train'][0])
    #print(wrapped_example)
    optimizer1,optimizer2,scheduler1,scheduler2=init_optimizer(args,data_loader,prompt_model)
    pbar = tqdm(data_loader, desc="training")
    for epoch in range(args.max_epochs):
        prompt_model.train()
        for  step, inputs in enumerate(pbar):  
            inputs = inputs.to(args.device)
           # input_batch = {key: batch[key] for key in batch if key in forward_keys}
            outputs = prompt_model.forward_without_verbalize(inputs)

            # if optimizer1 is not None:
            #     optimizer1.step()
            #     optimizer1.zero_grad()
            # if scheduler1 is not None:
            #     scheduler1.step()
            # if optimizer2 is not None:
            #     optimizer2.step()
            #     optimizer2.zero_grad()
            # if scheduler2 is not None:
            #     scheduler2.step()
        
        
   

def evaluate(args,prompt_model, dataloader, desc):
    if dataloader is None:
        raise ValueError("dataloader must not be None!")
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        inputs = inputs.to(args.device)
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc

def main(): 

    args=get_arg()
    logger=MyLogger(args)
    args.logger=logger

    #set_seed(args.seed)
    seed_torch(args.seed)
    if found_directml:
        args.device= torch_directml.device()
    else:
        args.device=torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    #Loader=InitDataLoad(args)

    ostype= platform.platform()
    if "Windows" in ostype:
        DIR="E:/transformers"
    else:
        DIR="/data/linux/transformers"
    model_name=DIR+os.sep+args.plm_name

    if "roberta" in args.plm_name:
        model_type="roberta"
    elif "albert" in args.plm_name:
        model_type="albert"
    elif "bert"  in args.plm_name:
        model_type="bert"
    else:
        model_type=None
    plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)
    dsinfo=load_data(args)

    if args.device != "cpu":
        info=torch.cuda.mem_get_info()
        base=24000000000 #gpu base memory
        ratio=round(info[0] /base,2)

   
        newbatch_s=int( dsinfo["batch_s"]*ratio)
        if newbatch_s < 1:
            newbatch_s=1
        dsinfo["batch_s"]= newbatch_s



    args.plm=plm
    args.tokenizer=tokenizer

    args.WrapperClass=WrapperClass
    model_config.args=args

   

    args.soft_token_num=1
    args.init_from_vocab = False 
    mytemplate = SoftTemplate(model=plm, tokenizer=args.tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab,text=r'{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"} .' )
    myverbalizer = SoftVerbalizer(tokenizer, model=plm, classes=dsinfo["class_labels"])
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
    prompt_model.to(args.device)
    
    support_sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=False)
    dsinfo['support'] = support_sampler(dsinfo['train'], seed=args.seed)

    train_dataloader = PromptDataLoader(dataset=dsinfo["support"], template=mytemplate, tokenizer=args.tokenizer,
    tokenizer_wrapper_class=args.WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
    batch_size=dsinfo["batch_s"],shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

    if args.dev_mode:
        test_dataloader = PromptDataLoader(dataset=dsinfo["test"], template=mytemplate, tokenizer=args.tokenizer,
        tokenizer_wrapper_class=args.WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
        batch_size=dsinfo["batch_s"],shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    else:
        test_dataloader=None

    #prompt_model.parallelize()
    run_train_dataset(args,prompt_model,dsinfo,mytemplate,train_dataloader)
    print("done!")



if __name__ == "__main__":
    main()
