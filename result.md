

CODE:  0.8802379202031678

COMMENT:  954161103693814

SATD:   0.9789536981358989

SMELL:   0.86



结果解释：

https://blog.csdn.net/Arctic_Beacon/article/details/115757581


----------------------------------------------------------------------------------------



************************************************** Running Dataset SMELL **************************************************
##Num of label words for each label: [50, 50]
INFO:codeprompt:-- real batch_size: 30
tokenizing: 0it [00:00, ?it/s]Token indices sequence length is longer than the specified maximum sequence length for this model (575 > 512). Running this sequence through the model will result in indexing errors
tokenizing: 350it [00:00, 555.12it/s]
INFO:codeprompt:-- classname: CodePromptSelfAttention
INFO:codeprompt:-- template text: | {"placeholder": "text_a"} In summary , it was {"mask"} .|
Test: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:46<00:00,  3.87s/it]
INFO:codeprompt:************************************************************************************************************************
INFO:codeprompt:acc: 0.86, precision: 0.8616666666666666, recall: 0.8565677478720957, f1score: 0.8582398598103802
INFO:codeprompt:************************************************************************************************************************
done!


************************************************** Running Dataset CODE **************************************************
##Num of label words for each label: [46, 35, 43, 41, 43, 49, 43, 38, 47, 36, 43, 43, 47, 50, 50, 44, 49, 41, 47]
INFO:codeprompt:-- real batch_size: 30
tokenizing: 0it [00:00, ?it/s]Token indices sequence length is longer than the specified maximum sequence length for this model (1232 > 512). Running this sequence through the model will result in indexing errors
tokenizing: 44889it [02:23, 313.75it/s]
INFO:codeprompt:-- classname: CodePromptSelfAttention
INFO:codeprompt:-- template text: | Just {"mask"} ! {"placeholder": "text_a"}|
Test: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1497/1497 [04:02<00:00,  6.17it/s]
INFO:codeprompt:************************************************************************************************************************
INFO:codeprompt:acc: 0.8802379202031678, precision: 0.8823223829889614, recall: 0.8809143791846639, f1score: 0.8814928845808406
INFO:codeprompt:************************************************************************************************************************


************************************************** Running Dataset SATD **************************************************
##Num of label words for each label: [50, 50]
INFO:codeprompt:-- real batch_size: 30
tokenizing: 3159it [00:02, 1083.68it/s]Token indices sequence length is longer than the specified maximum sequence length for this model (1582 > 512). Running this sequence through the model will result in indexing errors
tokenizing: 6652it [00:06, 1081.72it/s]
INFO:codeprompt:-- classname: CodePromptSelfAttention
INFO:codeprompt:-- template text: | Just {"mask"} ! {"placeholder": "text_a"}|
Test: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 222/222 [18:20<00:00,  4.96s/it]
************************************************************************************************************************
acc: 0.9789536981358989, precision: 0.9433028922714926, recall: 0.9225699764316987, f1score: 0.9326274874758083
************************************************************************************************************************






************************************************** Running Dataset COMMENT **************************************************
##Num of label words for each label: [50, 33, 46, 49, 32, 13, 43, 40, 49, 48, 46, 10, 19, 50, 44, 45]
INFO:codeprompt:-- real batch_size: 30
tokenizing: 2247it [00:02, 937.94it/s]
INFO:codeprompt:-- classname: CodePromptSelfAttention
INFO:codeprompt:-- template text: | It was {"mask"} . {"placeholder": "text_a"}|
Test: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [06:01<00:00,  4.82s/it]
************************************************************************************************************************
acc: 0.954161103693814, precision: 0.8965365322959542, recall: 0.8703635159536143, f1score: 0.8792984948700885
************************************************************************************************************************
