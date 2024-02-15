

import os


#file="verbalizer/CODE/ssize_20_num_50_tmpid_0_verbalizer.txt"
#file="verbalizer/CODE/ssize_60_num_200_tmpid_3_verbalizer.txt"
file="verbalizer/COMMENT/ssize_5_num_30_tmpid_0_verbalizer.txt"
with open(file,"r") as f:
    info=f.read().splitlines()
    common=set()
    bFirstRun=True
    for line in info:
        labelword=line.split(",")
        if len(common) ==0 and bFirstRun:
            common= set(labelword)
            bFirstRun=False
        else:
            common=common & set(labelword)

    print(len(common))