function run_codeprompt
{

# $1 dataset, $2 support_per_class, $3 per class, $4 shot ,$5 groupname  per class
#//microsoft/codebert-base, //microsoft/codebert-base-mlm, #microsoft/codebert-base-mlm  \
  CUDA_VISIBLE_DEVICES=$GPU  python codeprompt.py  \
            --plm_name  roberta-large    \
            --dataset $1  \
            --support_num $2   \
            --num_per_class  $3 \
            --shot $4   \
            --max_epochs 20 \
            --verbalizer none \
            --group_num $5 \
            --seed    $6 \
            --model_name $7 \
            --log_dir  $8 \
            --lr  $9    \
            --full_mode
}






GPU=1
LOG_DIR="logs/all_test"
#ALL_MODELS=`echo "self bundle ffn"`
ALL_MODELS=`echo "self"`

ALLDS=`echo "COMMENT SATD SMELL CODE"`
for DS in $ALLDS
do
#run_codeprompt $DS 60 100 1  8000  885 self $LOG_DIR  3e-05                 
run_codeprompt $DS 60 100 1  8000  885 self $LOG_DIR  4e-05                 
#run_codeprompt $DS 60 100 1  8000  885 self $LOG_DIR "--soft_prompt"
done

for DS in $ALLDS
do
run_codeprompt $DS 60 100 1  8000  885 self $LOG_DIR  3e-05                 
#run_codeprompt $DS 60 100 1  8000  885 self $LOG_DIR  4e-05                 
#run_codeprompt $DS 60 100 1  8000  885 self $LOG_DIR "--soft_prompt"
done