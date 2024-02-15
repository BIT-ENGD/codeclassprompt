import logging  
import time
import os 

LOGS_DIR="logs"


def get_time_string():
    t = time.localtime()
    str=("%d_%d_%d_%d_%d_%d"%(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec))
    return str

def get_logger_path(args):
    base_path=args.log_dir+os.sep

    if not os.path.exists(base_path):
        os.makedirs(base_path)


    dataset=args.dataset[:6]

    if args.model_name == "self":
        layernum=f"_layer_{args.layer_num}"
    else:
        layernum =""
    if args.appname != "main":
       log_file_name="%s_ds_%s_seed_%d_date_%s_type_%s_lr_%s_epoch_%d%s_model_%s.log"%(args.appname,args.dataset,args.seed,get_time_string(),"soft" if args.soft_prompt else "hard",str(args.lr),args.max_epochs,layernum,args.model_name)

    return base_path+log_file_name


   
class MyLogger(object):
    def __init__(self,args):
        logfile=get_logger_path(args)
        logging.basicConfig(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fHandler = logging.FileHandler(logfile)
        self.logger = logging.getLogger("codeprompt")
        fHandler.setFormatter(formatter)
        self.logger.addHandler(fHandler)
    
    def info(self,msg):
        self.logger.info(msg)
        #print(msg)
    def error(self,error):
        self.logger.error(error)
        #print(error)
    