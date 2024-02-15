function search_verbalizer
{

# $1 dataset, $2 support_per_class, $3 per class
#//microsoft/codebert-base, //microsoft/codebert-base-mlm, roberta-large
  CUDA_VISIBLE_DEVICES=$GPU  python verbalizer_search.py  \
            --plm_name  microsoft/codebert-base-mlm   \
            --dataset $1  \
            --support_num $2   \
            --num_per_class  $3
}





PERNUMER=`echo "30 50 100 150 200 250"`
GPU=0
AVS_LOG="logs/concept_all_seed_v2"
SUPPORT_NUM=`echo "5 10 20 30 40 50 60"`

DATASET=`echo "SMELL"`
for DS in $DATASET
do
    for PC_NUMBER in $PERNUMER
    do
        for SS_NUMBER in $SUPPORT_NUM
        do
            search_verbalizer $DS $SS_NUMBER  $PC_NUMBER
        done
    done

done 