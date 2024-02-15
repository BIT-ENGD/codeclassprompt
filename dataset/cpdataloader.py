from .cp_processor import *
import os
DATA_DIR="dataset"
#DATASET={"guesslang":{"dir":"/data/linux/src-det/files","ds_fn":GuessLang_load_data,"loader":DataLoader,"train":"train","valid":"valid","test":"test","batch_size":128,"extra":"guesslang/languages.json"}}
DATASET={
    "CODE":{"dir":"el_codebert","processor":CodeProcessor,"train":"CODE_train.csv","test":"CODE_test.csv","batch_s":7,"max_seq":480,"accum_steps":8,"cut_off":-1,"template_num":4},  # default batch_size 8, codebert: 30  320/8, 480/6
    "SMELL":{"dir":"el_codebert","processor":SmellProcessor,"train":"SMELL_train.csv","test":"SMELL_test.csv","batch_s":7,"max_seq":360,"accum_steps":8,"cut_off":-1,"template_num":4},
    "COMMENT":{"dir":"el_codebert","processor":CommentProcessor,"train":"COMMENT_train.csv","test":"COMMENT_test.csv","batch_s":7,"max_seq":480,"accum_steps":8,"cut_off":-1,"template_num":4},
    "SATD":{"dir":"el_codebert","processor":SatdProcessor,"train":"CODE_train.csv","test":"CODE_test.csv","batch_s":7,"max_seq":480,"accum_steps":8,"cut_off":-1,"template_num":4},
         }
ALL_DS=["CODE","SMELL","COMMENT","SATD"]
def load_data(args):

    ds=args.dataset
    dslist=ALL_DS
    DSINFO={}
    if ds not in DATASET:
        raise ValueError("not in ds list!")
    #for ds in dslist:
    info=DATASET[ds]
    path=os.sep+DATA_DIR+os.sep+info["dir"]
    dprocessor=info["processor"]
    DSINFO={}
    DSINFO["name"]=ds
    if not args.test_only:
        DSINFO['train'] =dprocessor().get_train_examples(f"{path}")
    DSINFO['test'] =dprocessor().get_test_examples(f"{path}")
    DSINFO['class_labels'] =dprocessor().get_labels()
    DSINFO['max_seq'] = info["max_seq"]
    DSINFO["batch_s"] = info["batch_s"]
    DSINFO["accum_steps"]=info["accum_steps"]
    DSINFO["template_num"]=info["template_num"]
    
    if info["cut_off"] == -1:
            if args.nocut :
                DSINFO["cut_off"] = 0
            else:
                DSINFO["cut_off"] = 0.5
    else:
           DSINFO["cut_off"] =  info["cut_off"]
    return DSINFO
