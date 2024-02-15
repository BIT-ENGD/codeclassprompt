function search_verbalizer
{

# $1 dataset, $2 support_per_class, $3 per class, $4 shot 
#//microsoft/codebert-base, //microsoft/codebert-base-mlm,
  CUDA_VISIBLE_DEVICES=$GPU  python fewshot.py  \
            --plm_name  roberta-large  \
            --dataset $1  \
            --support_num $2   \
            --num_per_class  $3 \
            --shot $4   \
            --max_epochs 10 \
            --dev_mode
}





PERNUMER=`echo "150 200 250"`
GPU=0
AVS_LOG="logs/concept_all_seed_v2"
SUPPORT_NUM=`echo "50 60"`

SHOT_NUM=`echo "150 200 250 300 350 400 350"`

DATASET=`echo "CODE"`
for DS in $DATASET
do
    for ST in $SHOT_NUM
    do
        for PC_NUMBER in $PERNUMER
        do
            for SS_NUMBER in $SUPPORT_NUM
            do
                search_verbalizer $DS $SS_NUMBER  $PC_NUMBER $ST
            done
        done

    done 

done