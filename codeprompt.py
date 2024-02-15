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

# def evaluate(args,prompt_model, dataloader, desc):
#     prompt_model.eval()
#     allpreds = []
#     alllabels = []
#     pbar = tqdm(dataloader, desc=desc)
#     for step, inputs in enumerate(pbar):
#         inputs = inputs.to(args.device)
#         logits = prompt_model(inputs,False)
#         labels = inputs['label']
#         alllabels.extend(labels.cpu().tolist())
#         allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
#     acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
#     return acc


def evaluate(args,prompt_model, dataloader, desc, lasteEva=False):
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
    if lasteEva:
        args.logger.info("*"*120)
        args.logger.info(f'acc: {acc}, precision: {precision}, recall: {recall}, f1score: {f1score}')
        args.logger.info("*"*120)
    return acc
    

def codepromt_one_dataset(dsinfo,args,label_word_path,template,tempid,model_type,model_name):

    plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)

    model_config.args=args


   
    # max ,mean, first
    if "zeroshot" == args.action:
        multi_handler="first"
    else:
        multi_handler=args.multi_handler
    args.logger.info("-- multi_handler_actual: "+multi_handler) 

    #class_labels=dsinfo["class_labels"]
    #load first label words as classname

    fake_class_labels=load_first_label_word_as_classname(label_word_path)
    class_labels=fake_class_labels
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=dsinfo["cut_off"],max_token_split=args.max_token_split,multi_token_handler=multi_handler).from_file(label_word_path)

    if args.soft_prompt:
        mytemplate = SoftTemplate(model=plm,tokenizer=tokenizer,text=template)
    else:
        mytemplate = ManualTemplate(tokenizer=tokenizer,text= template ) 

    #  CodePromptSelfAttention(self),CodePromptBundleAttention(bundle),CodePromptFFN (ffn)

    if args.model_name == "ffn": 
        prompt_model = CodePromptFFN(plm=plm,template=mytemplate, verbalizer=myverbalizer,freeze_plm=False, plm_eval_mode=args.plm_eval_mode,args=args)
    elif args.model_name == "self":
        prompt_model = CodePromptSelfAttention(plm=plm,template=mytemplate, verbalizer=myverbalizer,freeze_plm=False, plm_eval_mode=args.plm_eval_mode,args=args)
    elif args.model_name == "bundle":
        prompt_model = CodePromptBundleAttention(plm=plm,template=mytemplate, verbalizer=myverbalizer,freeze_plm=False, plm_eval_mode=args.plm_eval_mode,args=args)


    prompt_model=prompt_model.to(args.device)
    ## support set for fewshot
    # 

    args.logger.info("-- classname: {}".format(prompt_model.__class__.__name__))
    args.logger.info("-- template text: |{}|".format(template))




    batch_size= math.floor(dsinfo["batch_s"]/args.shot) *args.shot

    args.logger.info("-- real batch_size: {}".format(batch_size))
    decoder_max_length=3
    train_dataloader = PromptDataLoader(dataset=dsinfo["support"], template=mytemplate, tokenizer=tokenizer,
                                 tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=decoder_max_length,
                                  batch_size=batch_size,shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                 truncate_method="tail")

    #dsinfo["batch_s"]
    validation_dataloader = PromptDataLoader(dataset=dsinfo["validation"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=decoder_max_length,
        batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    
    if not args.dev_mode:
        test_dataloader = PromptDataLoader(dataset=dsinfo["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=decoder_max_length,
        batch_size=dsinfo["batch_s"],shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    else:
        test_dataloader=None


    loss_func = torch.nn.CrossEntropyLoss()
    optimizer,scheduler=build_optimizer_ex(args,prompt_model,dsinfo["batch_s"],args.max_epochs,train_dataloader)

 
    best_val_acc=0
    lrs=[]
    model_save_path=None
    all_models=[]
    for epoch in range(args.max_epochs):
        prompt_model.train()
        pbr=tqdm(train_dataloader,desc="training epoch {}".format(epoch))
        with torch.autograd.set_detect_anomaly(True):
            for step, inputs in enumerate(pbr):
                inputs = inputs.to(args.device)

                logits = prompt_model(inputs)
                labels = inputs['label']
                
                #indexs=[]
                # for index in range(0,inputs["label"].size(0), args.shot):
                #     indexs.append(index)
                # labels=torch.index_select(inputs["label"],0,torch.tensor(indexs).to(args.device))

                loss = loss_func(logits, labels)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            
                
                optimizer.step()
                scheduler.step()
                lrs.append(scheduler.get_last_lr())
                optimizer.zero_grad()


        val_acc = evaluate(args,prompt_model, validation_dataloader, desc="Valid")
        args.logger.info("Epoch {}, val_acc {}".format(epoch, val_acc))
        if val_acc>=best_val_acc:
            model_save_path = get_ckpt_name(args,val_acc,epoch,tempid)
            all_models.append(model_save_path)
            torch.save(prompt_model.state_dict(),f"ckpts/{model_save_path}.ckpt")
            best_val_acc = val_acc
        
            

 

    if test_dataloader is not None and model_save_path is not None:
        prompt_model.load_state_dict(torch.load(f"ckpts/{model_save_path}.ckpt"))
        prompt_model = prompt_model.cuda()
        test_acc = evaluate(args,prompt_model, test_dataloader, desc="Test",lasteEva=True)
        args.logger.info("Final test_acc: {},model_name: {}".format(test_acc,model_save_path))

        #remove redundant models

    else:
        test_acc=0

    for model_path in all_models:
        if model_path != model_save_path:  # best model
            path=f"ckpts/{model_path}.ckpt"
            if os.path.exists(path):
               os.remove(path)
       
    return test_acc


def takelabel(elem):
    return elem.label
def sort_sample_by_class(info):
    return info.sort(key=takelabel)

def sort_sample_by_class_package(info,shot):

    bags=[]
    #random.shufflt(list)
    realshot=shot

    total=len(info)
    sort_sample_by_class(info)
    for index in range(0,total,realshot):
        subbag=info[index:(index+realshot)]
        random.shuffle(subbag)
        bags.append(subbag)
    random.shuffle(bags)
    all_list=[]
    for item in bags:
        all_list.extend(item)
   
    return all_list

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

    dsinfo=load_data(args)


    if args.device == "gpu" or args.device == "dml":
        info=args.device.mem_get_info()
        base=24000000000 #gpu base memory
        ratio=round(info[0] /base,2)
        newbatch_s=int( dsinfo["batch_s"]*ratio)
        if newbatch_s < 1:
            newbatch_s=1
        dsinfo["batch_s"]= newbatch_s






    if args.max_length> 0:
        dsinfo["max_seq"] = args.max_length


    if args.batch_size >0:
        dsinfo["batch_s"]= args.batch_size


    args.logger.info("-- Batch_size:{}".format(dsinfo["batch_s"]))
    args.logger.info("-- Max_seq_length:{}".format(dsinfo["max_seq"]))
    args.logger.info("-- Layer_Number:{}".format(args.layer_num))

    output_args(args,dsinfo)


    

    # bag_attention, need to sort the training data
    if args.full_mode:

        dsinfo["validation"]=dsinfo["test"]
        dsinfo['support']= dsinfo['train']

    else:
        sample_number_per_class= args.group_num *args.shot
        sampler = FewShotSampler(num_examples_per_label=sample_number_per_class, also_sample_dev=True, num_examples_per_label_dev=sample_number_per_class)
        dsinfo['support'], dsinfo['validation'] = sampler(dsinfo['train'], seed=args.seed)
        # if args.shot >1 and args.model_name == "attention":
        #     dsinfo['support']=sort_sample_by_class_package(dsinfo['support'],args.shot)

    if args.template_id == -1:
        all_template_id=range(dsinfo["template_num"])
    else:
        all_template_id=[args.template_id]

    templates=load_template(args)
    for template_id in all_template_id:
        label_word_path=get_label_words_path(args,template_id)
        #(dsinfo,args,label_words_path,template,tempid):
        
        codepromt_one_dataset(dsinfo,args,label_word_path,templates[template_id],template_id,model_type,model_name)
        #fewshot_one_dataset(dsinfo,args,label_word_path,templates[template_id],template_id)
        print("done!")



if __name__ == "__main__":
    main()
