import numpy as np
import random
import torch 
import os
import pickle as pk
import datetime

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


'''


def set_seed(seed:Optional[int] = None):
    """set seed for reproducibility

    Args:
        seed (:obj:`int`): the seed to seed everything for reproducibility. if None, do nothing.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Global seed set to {seed}")

'''

DATASET_PATH="dataset"




def cache_data(data,path):
	with open(path,"wb") as f:
		pk.dump(data,f)
	

def load_cache(path):
	with open(path,"rb") as f:
		return pk.load(f)

def output_args(args,dsinfo):
	info=args.logger.info
	param=args.__dict__
	exclude_list=["model","plm","lm_head","tokenizer","config","logger","wrapperclass","DATASET"]
	
	for paraname in param:
		if paraname not in exclude_list:
			info("-- {} : {}".format(paraname,param[paraname]))


def get_template_path(args,type="manual"):
	if args.soft_prompt:
		type ="soft"

	return "scripts/{}_template.txt".format(type)

def get_label_words_path(args,template_id):
	return "verbalizer/{}/ssize_{}_num_{}_tmpid_{}_verbalizer.txt".format(args.dataset,args.support_num,args.num_per_class,template_id)

def load_template(args):
    template_path=get_template_path(args)
    with open(template_path,"r") as f:
         alltemp=f.readlines()


    newtemp=[" "+t.strip() for t in alltemp]
        
    return newtemp


def load_first_label_word_as_classname(label_path):
	fake_class_names=list()
	with open(label_path,"r") as f:
		line=f.read().splitlines()
		for single in line:
			single=single.split(",")
			#fake_class_names.append(single[0])
			fake_class_names.append("")

	return fake_class_names


def get_ckpt_name(args,acc,epoch,tmpid):

	date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	plm_name=args.plm_name.replace("/","-")
	return "name_{}_date_{}_seed_{}_epoch_{}_acc_{}_plm_{}_model_{}_type_{}_tmpid_{}".format(args.dataset,date,args.seed,epoch,round(acc,6),plm_name,args.model_name,"soft" if args.soft_prompt else "hard",tmpid)

def get_weight_file_name(args):
	pass