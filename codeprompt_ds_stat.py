import torch
from openprompt.prompts import KnowledgeableVerbalizer,ManualVerbalizer,ManualTemplate,SoftTemplate
from util import *
from dataset import *
from model import *
from torch.utils.tensorboard.writer import SummaryWriter
from openprompt.utils.reproduciblity import set_seed
from torch.optim import AdamW 
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt import PromptForClassification
#from transformers.data.metrics import acc_and_f1, simple_accuracy
from sklearn.metrics import f1_score
from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from  tqdm import tqdm
import platform
import math
import os
import re
from importlib import util as imp_util
import random
from transformers import  get_linear_schedule_with_warmup
try:
    imp_util.find_spec('torch_directml')
    found_directml = True
    import torch_directml
except ImportError:
    found_directml = False


from contextualize_calibration import *


'''
abel:
就是你计算有多少个batch，然后总的step就是batch乘以epoch，你这里好像是直接设置了一个max值，可能这个数比最终的小了


abel:
num warmup step一般是总的step的十分之一，num trainingup step就是上述那种计算方式，你这里给的是max step 可能比实际的小很多，所以之后的step学习率减小到0了，参数不更新了，acc就不变了


'''


MODEL_PATH=[
    "CODE",
    "SMELL",
    "COMMENT",
    "SATD"
]
def build_optimizer_ex(args,model,batch_size,epoch,train_dataloader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    len_dataset =len(train_dataloader.wrapped_dataset)
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
    warm_up_ratio = args.warmup_ratio # 定义要预热的step ，  warm_up_ratio * total_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

   # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    return optimizer,scheduler

def evaluate(args,prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        inputs = inputs.to(args.device)
        logits = prompt_model(inputs,False)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    precision, recall, f1score = precision_recall_fscore_support(alllabels, allpreds,average='macro')[:3]
    acc = accuracy_score(alllabels, allpreds)
    args.logger.info("*"*120)
    args.logger.info(f'acc: {acc}, precision: {precision}, recall: {recall}, f1score: {f1score}')
    args.logger.info("*"*120)
    return acc


def evaluate_time(args,prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm(dataloader, desc=desc)
 
    for step, inputs in enumerate(pbar):
        inputs = inputs.to(args.device)
        T1 =time.perf_counter()
        for i in range(1000):
            logits = prompt_model(inputs,False)
        T2 =time.perf_counter()
        #print('Step %d 程序运行时间:%s毫秒' % (step,(T2 - T1)*1000))

    return (T2 - T1)*1000

def parse_model_path(model_path):


    result=re.findall("[0-9]+",model_path.strip()) #查找 seed 值 
    
    if len(result)>10:
        return int(result[10])
    else:
        return -1



def main(): 

    args=get_arg()
    
    args.appname="inference"
    logger=MyLogger(args)
    args.logger=logger

    set_seed(args.seed)
    if found_directml:
        args.device= torch_directml.device()
    else:
        args.device=torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ostype= platform.platform()
    if "Windows" in ostype:
        DIR="E:/transformers"
    else:
        DIR="/data/linux/transformers"
    model_name=DIR+os.sep+args.plm_name

    if "roberta" in args.plm_name or "codebert" in args.plm_name:
        model_type="roberta"
    elif "albert" in args.plm_name:
        model_type="albert"
    elif "bert"  in args.plm_name:
        model_type="bert"
    else:
        model_type=None

      

    # bag_attention, need to sort the training data
    # if args.full_mode:

    # dsinfo["validation"]=dsinfo["test"]
    # dsinfo['support']= dsinfo['train']

    # else:
    #     sample_number_per_class= args.group_num *args.shot
    #     sampler = FewShotSampler(num_examples_per_label=sample_number_per_class, also_sample_dev=True, num_examples_per_label_dev=sample_number_per_class)
    #     dsinfo['support'], dsinfo['validation'] = sampler(dsinfo['train'], seed=args.seed)
    #     # if args.shot >1 and args.model_name == "attention":
        #     dsinfo['support']=sort_sample_by_class_package(dsinfo['support'],args.shot)

    # if args.template_id == -1:
    #     all_template_id=range(dsinfo["template_num"])
    # else:
    #     all_template_id=[args.template_id]

    if args.dataset != "ALLDS":
        DSLIST=[args.dataset]
    else:
        DSLIST=MODEL_PATH
    
    for dataset in DSLIST:
        TRAIN_STAT={}
        TEST_STAT={}
        args.dataset=dataset
        dsinfo=load_data(args)


        if args.device != "cpu":
            info=torch.cuda.mem_get_info()
            base=24000000000 #gpu base memory
            ratio=round(info[0] /base,2)
            newbatch_s=int( dsinfo["batch_s"]*ratio)
            if newbatch_s < 1:
                newbatch_s=1
            dsinfo["batch_s"]= newbatch_s


        # args.logger.info("-- Batch_size:{}".format(dsinfo["batch_s"]))
        # args.logger.info("-- Max_seq_length:{}".format(dsinfo["max_seq"]))
        # args.logger.info("-- Layer_Number:{}".format(args.layer_num))

        for item in dsinfo["train"]:
            if item.label not in TRAIN_STAT:
                TRAIN_STAT[item.label]=0

            TRAIN_STAT[item.label]+=1

        for item in dsinfo["test"]:
            if item.label not in TEST_STAT:
                TEST_STAT[item.label]=0

            TEST_STAT[item.label]+=1
        
        TRAIN_STAT=sorted(TRAIN_STAT.items(),key=lambda s:s[0])
        TEST_STAT=sorted(TEST_STAT.items(),key=lambda s:s[0])
        
        
        print("*"*30,"dataset:",dataset,"*"*24)
        print("label : count")
        print("*"*30,"train set","*"*30)
        for item in TRAIN_STAT:
            print(item[0],":\t",item[1])
        print("*"*30,"test set","*"*30)
        for item in TEST_STAT:
            print(item[0],":\t",item[1])
        print("-"*70)
      
    print("done!")



if __name__ == "__main__":
    main()
