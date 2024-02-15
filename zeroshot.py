import torch
from openprompt.prompts import SoftVerbalizer,KnowledgeableVerbalizer,ManualVerbalizer,ManualTemplate
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


from contextualize_calibration import *

def run_once(args,prompt_model, dataloader, desc):
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
    
def zeroshot_one_dataset(dsinfo,args,label_words_path,template,tempid):

    tokenizer=args.tokenizer
    plm=args.plm
    WrapperClass=args.WrapperClass

   
    # max ,mean, first
    if "zeroshot" == args.action:
        multi_handler="first"
    else:
        multi_handler=args.multi_handler
    args.logger.info("-- multi_handler_actual: "+multi_handler) 

    class_labels=dsinfo["class_labels"]
    #load first label words as classname

    fake_class_labels=load_first_label_word_as_classname(label_words_path)
    class_labels=fake_class_labels
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=dsinfo["cut_off"],max_token_split=args.max_token_split,multi_token_handler=multi_handler).from_file(label_words_path)
    #elif args.verbalizer == "manual":
    #    myverbalizer = ManualVerbalizer(tokenizer, classes=dsinfo["class_labels"]).from_file(label_words_path)

    mytemplate = ManualTemplate(tokenizer=tokenizer,text= template ) 
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode).to(args.device)
    if args.word_calibration:
        for example in dsinfo['support']:
            example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dsinfo["support"], template=mytemplate, tokenizer=tokenizer,
                                tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
                                 batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                truncate_method="tail")
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(dsinfo["class_labels"]))]
        str_num=[str(num) for num in org_label_words_num]
        args.logger.info("label words before calibration: ["+",".join(str_num)+"]")

        myrecord = ""
        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
       # print("the calibration logits is", cc_logits)
        myrecord += "Phase 1 {}\n".format(org_label_words_num)

        myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))  #低频 label words去除，对应的： Frequency Refinement. 方法，去除低频label words
        
    new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(dsinfo["class_labels"]))]
    str_num=[str(num) for num in new_label_words_num]
    args.logger.info("label words after calibration: ["+",".join(str_num)+"]")



        
    # with open(fined_label_path,"w") as f:
    #     for classid,labels in enumerate(myverbalizer.label_words):
    #         labels=",".join(labels)+"\n"
    #         f.write(labels)
            


    test_dataloader = PromptDataLoader(dataset=dsinfo["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
        batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    label_data=dsinfo["class_labels"]
    pbar = tqdm(test_dataloader)
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(pbar):
        
        inputs = inputs.to(args.device)
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    args.logger.info(metrics_calculate(allpreds,alllabels,label_data,True))
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    args.logger.info("dataset:{}, acc {} ,template [{}]".format(dsinfo["name"],acc,template  ))
    f1,recall,precision=metrics_calculate_ex(allpreds,alllabels,label_data,True)
    return [f1,recall,precision,acc,alllabels,allpreds]

 



def main(): 

    args=get_arg()
    logger=MyLogger(args)
    args.logger=logger

    set_seed(args.seed)
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

    output_args(args,dsinfo)

    #mytemplate = ManualTemplate(model=plm, tokenizer=args.tokenizer, initialize_from_vocab=args.init_from_vocab,text=r'{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"} .' )
    #myverbalizer = SoftVerbalizer(tokenizer, model=plm, classes=dsinfo["class_labels"])


    
    support_sampler = FewShotSampler(num_examples_total=args.cali_support_num, also_sample_dev=False)
    dsinfo['support'] = support_sampler(dsinfo['train'], seed=args.seed)

    # train_dataloader = PromptDataLoader(dataset=dsinfo["support"], template=mytemplate, tokenizer=args.tokenizer,
    # tokenizer_wrapper_class=args.WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
    # batch_size=dsinfo["batch_s"],shuffle=True, teacher_forcing=False, predict_eos_token=False,
    # truncate_method="tail")

    # if args.dev_mode:
    #     test_dataloader = PromptDataLoader(dataset=dsinfo["test"], template=mytemplate, tokenizer=args.tokenizer,
    #     tokenizer_wrapper_class=args.WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
    #     batch_size=dsinfo["batch_s"],shuffle=True, teacher_forcing=False, predict_eos_token=False,
    #     truncate_method="tail")
    # else:
    #     test_dataloader=None

    if args.template_id == -1:
        all_template_id=range(dsinfo["template_num"])
    else:
        all_template_id=[args.template_id]

    templates=load_template(args)
    for template_id in all_template_id:
        label_word_path=get_label_words_path(args,template_id)
        #(dsinfo,args,label_words_path,template,tempid):
        zeroshot_one_dataset(dsinfo,args,label_word_path,templates[template_id],template_id)
        print("done!")



if __name__ == "__main__":
    main()
