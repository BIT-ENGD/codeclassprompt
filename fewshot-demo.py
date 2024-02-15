import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup,get_linear_schedule_with_warmup  # use AdamW is a standard practice for transformer
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
try:
    imp_util.find_spec('torch_directml')
    found_directml = True
    import torch_directml
except ImportError:
    found_directml = False





def prompt_initialize(args,verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.to(args.device)
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


def evaluate(args,prompt_model, dataloader, desc):
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



def build_optimizer(args,prompt_model,train_dataloader,myverbalizer):
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters1 = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
            optimizer1,
            num_warmup_steps=0, num_training_steps=tot_step)
        
    if args.verbalizer == "soft":

        optimizer_grouped_parameters2 = [
            {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
            {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
        ]
        optimizer2 = AdamW(optimizer_grouped_parameters2)

        scheduler2 = get_linear_schedule_with_warmup(
            optimizer2,
            num_warmup_steps=0, num_training_steps=tot_step)

    elif args.verbalizer == "auto":
        prompt_initialize(myverbalizer, prompt_model, train_dataloader)
        optimizer2 = None
        scheduler2 = None

    elif args.verbalizer == "kpt" or args.verbalizer == "concept":
        optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.lr)
        scheduler2 = None

    elif args.verbalizer == "manual":
        optimizer2 = None
        scheduler2 = None

    
    return optimizer1,optimizer2,scheduler1,scheduler2


def run_fewshot_one_dataset(args,dsinfo,template,tmp_id,label_words_path):
    mytemplate = ManualTemplate(tokenizer=args.tokenizer,text= template ) 
   
        # (contextual) calibration

    if args.verbalizer in ["kpt","manual","concept"]:
        if args.word_calibration or args.filter != "none":
            support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
            dsinfo['support'] = support_sampler(dsinfo['train'], seed=args.seed)

            # for example in dataset['support']:
            #     example.label = -1 # remove the labels of support set for clarification
            support_dataloader = PromptDataLoader(dataset=dsinfo["support"], template=mytemplate, tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
                batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")

    class_labels=dsinfo["class_labels"]
    multi_handler =args.multi_handler
    if args.verbalizer == "manual":
        myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(label_words_path)
    else:
        myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=dsinfo["cut_off"],max_token_split=args.max_token_split,multi_token_handler=multi_handler).from_file(label_words_path)

    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)

    prompt_model=  prompt_model.to(args.device)

    if args.verbalizer in ["kpt","manual","concept"]:
        if args.word_calibration or args.filter != "none":
            org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
            from contextualize_calibration import calibrate
            # calculate the calibration logits
            cc_logits = calibrate(prompt_model, support_dataloader)
            print("the calibration logits is", cc_logits)
            print("origial label words num {}".format(org_label_words_num))

        if args.word_calibration: 
            myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
            new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
            print("After filtering, number of label words per class: {}".format(new_label_words_num))


    
        if args.filter == "tfidf_filter":
            tfidf_filter(myverbalizer, cc_logits, class_labels)
        elif args.filter == "none":
            pass
        else:
            raise NotImplementedError

    #prepare fewshot learning

    # to obtain training/validation samples

    sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
    dsinfo['train_r'], dsinfo['validation'] = sampler(dsinfo['train'], seed=args.seed)

    train_dataloader = PromptDataLoader(dataset=dsinfo["train_r"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
    batch_size=dsinfo["batch_s"],shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
#
    validation_dataloader = PromptDataLoader(dataset=dsinfo["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
    batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
    test_dataloader = PromptDataLoader(dataset=dsinfo["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
    batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer1,optimizer2,scheduler1,scheduler2=build_optimizer(args,prompt_model,train_dataloader,myverbalizer)


    log_loss = 0
    best_val_acc = 0
    for epoch in range(args.max_epochs):
        tot_loss = 0
        prompt_model.train()
        with torch.autograd.set_detect_anomaly(True):
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(args.device)
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            
                
                optimizer1.step()
                scheduler1.step()
            
                if optimizer2 is not None:
                    optimizer2.step()
                    optimizer2.zero_grad()
                if scheduler2 is not None:
                    scheduler2.step()

                optimizer1.zero_grad()

        val_acc = evaluate(args,prompt_model, validation_dataloader, desc="Valid")
        if val_acc>=best_val_acc:
            torch.save(prompt_model.state_dict(),f"ckpts/{this_run_unicode}.ckpt")
            best_val_acc = val_acc
        print("Epoch {}, val_acc {}".format(epoch, val_acc), flush=True)

    prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}.ckpt"))
    prompt_model = prompt_model.cuda()
    test_acc = evaluate(args,prompt_model, test_dataloader, desc="Test")
    return test_acc 
    #print("===== final acc: %f ===="%(test_acc) )

if __name__ == "__main__":
    args=get_concept_args()

    args.appname="fewshot"
    
    logger=MyLogger(args)
    args.logger=logger

    this_run_unicode = str(random.randint(0, 1e10))

    set_seed(args.seed)
    dsinfo=prepare_data(args,False)

    if args.device != "cpu":
        info=torch.cuda.mem_get_info()
        base=24000000000 #gpu base memory
        ratio=round(info[0] /base,2)

        for ds in dsinfo:
            newbatch_s=int( DATASET[ds]["batch_s"]*ratio)
            if newbatch_s < 1:
                newbatch_s=1
            dsinfo[ds]["batch_s"]= newbatch_s

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
    ds=args.dataset
    output_args(args,dsinfo[ds])
    args.tokenizer = tokenizer
    args.plm=plm
  
    templates=load_template(args,dsinfo[ds])

  
    if args.device == "cuda":
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device= torch.device("cpu")


    range_end  = len(templates)
    range_start = 0

    if args.template_id > -1:
        range_start=args.template_id
        range_end =args.template_id + 1

    class_labels= CLASSNAME[ds] 

    for tmp_index in range(range_start,range_end):

        if args.verbalizer == "concept":  #["concept","manual","kpt","soft","auto"]
            label_words_path=get_verbalopt_label_path(args,ds,tmp_index)
        elif "kpt" == args.verbalizer : #kpt
            label_words_path="scripts"+os.sep+dsinfo[ds]["name"]+os.sep+"knowledgeable_verbalizer."+dsinfo[ds]["script_format"]
        elif "manual" == args.verbalizer :  #pt 
            label_words_path="scripts"+os.sep+dsinfo[ds]["name"]+os.sep+"manual_verbalizer."+dsinfo[ds]["script_format"]

        test_acc=run_fewshot_one_dataset(args,dsinfo[ds],templates[tmp_index],tmp_index,label_words_path)
        args.logger.info("===========test acc of template id %d is: %f"%(tmp_index,test_acc))
