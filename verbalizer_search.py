import torch

from util import *
from dataset import *
from model import *
from torch.utils.tensorboard.writer import SummaryWriter
from openprompt.utils.reproduciblity import set_seed
from torch.optim import AdamW 
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.prompts import SoftVerbalizer,ManualVerbalizer,ManualTemplate
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

def filter_words(tokens: List[str], word_counts=None, max_words: int = -1):
    """
    Given a list of tokens, return a reduced list that contains only tokens from the list that correspond
    to actual words and occur a given number of times.
    :param tokens: the list of tokens to filter
    :param word_counts: a dictionary mapping words to their number of occurrences
    :param max_words: if set to a value >0, only the `max_words` most frequent words according to `word_counts` are kept
    :return: the filtered list of tokens
    """
    tokens = (word for word in tokens if word[0] == 'Ġ' and len([char for char in word[1:] if char.isalpha()]) >= 2)
    if word_counts and max_words > 0:
        tokens = sorted(tokens, key=lambda word: word_counts[word[1:]], reverse=True)[:max_words]
    return tokens



def get_word_to_id_map(tokenizer: PreTrainedTokenizer, word_counts=None, max_words: int = -1):
    """
    Return a mapping from all tokens to their internal ids for a given tokenizer
    :param tokenizer: the tokenizer
    :param word_counts: a dictionary mapping words to their number of occurrences
    :param max_words: if set to a value >0, only the `max_words` most frequent words according to `word_counts` are kept
    :return:
    """

    words = filter_words(tokenizer.encoder.keys(), word_counts, max_words) #过滤掉不需要的词
    word2id = {word[1:]: tokenizer.convert_tokens_to_ids(word) for word in words} #将词转成id
   # logger.info(f"There are {len(word2id)} words left after filtering non-word tokens")
    return word2id
def verbalizer_optimizer_ex(dsinfo,args,label_words_path,template,tempid):

    tokenizer=args.tokenizer
    plm=args.plm
    WrapperClass=args.WrapperClass

    verbalop_label_path=label_words_path
    
    myverbalizer = ManualVerbalizer(tokenizer, classes=dsinfo["class_labels"]) # .from_file(label_words_path)  # placeholder only, no effectiveness
    mytemplate = ManualTemplate(tokenizer=tokenizer,text= template ) 
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode).to(args.device)
    

    test_dataloader = PromptDataLoader(dataset=dsinfo["support"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=dsinfo["max_seq"], decoder_max_length=3,
        batch_size=dsinfo["batch_s"],shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    label_data=dsinfo["class_labels"]
    pbar = tqdm(test_dataloader)

    all_logits = []
    
    
    
   
    all_logits=list()
    all_labels=list()
    expected=dict()
    for step, batch in enumerate(pbar):
        
        batch = batch.to(args.device)
        logits = prompt_model.forward_without_verbalize(batch)
        labels = batch['label']
        all_logits.extend(logits.detach().to("cpu"))
        all_labels.extend(labels.detach().to("cpu"))
    all_labels=torch.hstack(all_labels)
    all_logits=torch.vstack(all_logits).numpy()
# to adapt data format.
    max_words=10000

    all_labels=all_labels 
    expected={}
    for classid,item in enumerate(label_data):
        expected[item] = (all_labels == classid).to(torch.int).numpy() # support size 200, 这是一个掩码，体现属于本类的标志，属于本类为1,否则 为0


    word2idx=get_word_to_id_map(args.tokenizer,None,max_words) #load_word2idx(args)
    verbalizer_optimizer=VerbalOptimizerEx(word2idx, label_data,all_logits,expected)

    num_candidates =len(word2idx)
    words_per_label=args.num_per_class
    verbalizers=verbalizer_optimizer.find_verbalizer(
                num_candidates=num_candidates,
                words_per_label=words_per_label,
                normalize= not args.imbalance,  #True, #args.normalize,
                score_fct="llr"
            )

    with open(verbalop_label_path,"w") as f:
        for classname in verbalizers:
            f.write(",".join(verbalizers[classname])+"\n")
    return verbalizers

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
    elif "codebert" in args.plm_name:
        model_type="roberta"
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
       
    
    support_sampler = FewShotSampler(num_examples_per_label=args.support_num, also_sample_dev=False)
    dsinfo['support'] = support_sampler(dsinfo['train'], seed=args.seed)
    template=get_template_path(args)


    templates=load_template(args)
    for template_id in range(dsinfo["template_num"]):
        label_words_path=get_label_words_path(args,template_id)
        verbalizer_optimizer_ex(dsinfo,args,label_words_path,templates[template_id],template_id)
    print("done!")

if __name__ == "__main__":
    main()
