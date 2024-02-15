import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from numpy import *




def addComma(num):
    to_str=str(num) #转换成字符串
    count=0 #循环计数
    sumstr='' #待拼接的字符串
    for one_str in to_str[::-1]: #注意循环是倒着输出的
        count += 1 #计数
        if count %3==0 and count != len(to_str): #如果count等于3或3的倍数并且不等于总长度
            one_str = ',' + one_str # 当前循环的字符串前面加逗号
            sumstr = one_str + sumstr #拼接当前字符串
        else:
            sumstr = one_str + sumstr #正常拼接字符串
    return sumstr #返回拼接的字符串


def testnumber(name):
    df = pd.read_csv(f"dataset/el_codebert/{name}_test.csv")
    code_list = df["label"].tolist()
    print("test number:",addComma(len(code_list)))
    
def getBili(num, demo_list):
    s = 0
    for i in range(len(demo_list)):
        if(demo_list[i] < num):
            s += 1
    print('<'+str(num)+'比例为'+str(s/len(demo_list)))

def statistic(name,fldname,fldlabel):
    print("*"*50,name,"*"*50)
    df = pd.read_csv(f"dataset/el_codebert/{name}_train.csv")
    code_list = df[fldname].tolist()
    label_list = df[fldlabel].tolist()
    print("Train Number:",addComma(len(label_list)))
    testnumber(name)
    code_len_list = [len(str(code).split()) for code in code_list]
    print("class number:",len(list(set(label_list))))
    #
    # count = 0
    # for i in range(len(label_list)):
    #     if(label_list[i] == 0):
    #         count+=1
    # print(count, len(label_list)-count)





    b = mean(code_len_list)
    c = median(code_len_list)
    counts = np.bincount(code_len_list)
    d = np.argmax(counts)
    print('平均值'+str(b))
    print('众数'+str(d))
    print('中位数'+str(c))

    getBili(32,code_len_list)
    getBili(64,code_len_list)
    getBili(128,code_len_list)
    getBili(256,code_len_list)
    getBili(300,code_len_list)
    # getBili(512,code_len_list)



statistic("SMELL","code","label")
statistic("CODE","code","label")
statistic("SATD","text","label")
statistic("COMMENT","text","label")



