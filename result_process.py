import numpy as np
import os
import glob
import re


ACCURACY="ACCURACY"
PRECISION="PRECISION"
RECALL="RECALL"
F1_SCORE="F1_SCORE"

VALUE_TYPE={"ACCURACY":0,"PRECISION":1,"RECALL":2,"F1_SCORE":3}

def parse_result(filepath):

    acc=0
    prec=0
    recall=0
    f1=0
    with open(filepath) as f:
        content=f.readlines()
        for line in content:
            result=re.findall("acc:+",line) #查找 seed 值
            if len(result)>0:
                result=re.findall("[0-9\.]+",line)
                acc =result[7]
                prec =result[8]
                recall =result[9]
                f1 =result[11]
                break




    return acc,prec,recall,f1
def get_result(oridir):

    curdir=os.getcwd()
    os.chdir(oridir)
    ALL_RESULT=dict()
    for file in glob.glob("*.log"):
        fullpath=oridir+os.sep+file
        char_result= re.findall("[a-z]+",file.strip(),re.IGNORECASE)
        dataset=char_result[2]
        if( char_result[11] == "model"):
            model=char_result[12]
        else:
            model=char_result[11]
        
        if model == "self":
            result=re.findall("[0-9]+",file.strip()) #查找 seed 值 
            layernum=int(result[10])
        else:
            layernum =-1
     
        acc,prec,recall,f1=parse_result(file)
        if dataset not in ALL_RESULT:
            ALL_RESULT[dataset]=dict()
        
            

        ALL_RESULT[dataset][layernum]=[acc,prec,recall,f1]


    
    os.chdir(curdir)
       
    return ALL_RESULT

def show_result(dirname,type="ACCURACY"):

    print("\n#","="*50,type,"="*50)
    ALL_RESULT=get_result(dirname)

    type=VALUE_TYPE[type]
    for ds in ALL_RESULT:
        print("#","*"*50,ds,"*"*50,"\n{}=[".format(ds),end="")
        valuelist=dict()
        for layer in range(-1,max(ALL_RESULT[ds].keys())+1):
            value=ALL_RESULT[ds][layer]
            if layer >-1:
                print("%6.3f"% (round(float(value[type]),5)*100),end="")
            if layer> -1 and layer<12:
                print(",",end="")
        print("]")
    

   




if __name__ == "__main__":

    
    show_result("logs/ablation_3090")
    show_result("logs/ablation_3090",PRECISION)
    show_result("logs/ablation_3090",RECALL)
    show_result("logs/ablation_3090",F1_SCORE)

    print("#===done===")