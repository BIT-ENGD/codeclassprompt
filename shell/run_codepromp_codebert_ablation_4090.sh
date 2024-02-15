function run_codeprompt
{

# $1 dataset, $2 support_per_class, $3 per class, $4 shot ,$5 groupname  per class
#//microsoft/codebert-base, //microsoft/codebert-base-mlm, #microsoft/codebert-base-mlm  \
  CUDA_VISIBLE_DEVICES=$GPU  python codeprompt.py  \
            --plm_name  microsoft/codebert-base-mlm    \
            --dataset $1  \
            --verbalizer none \
            --seed    $2 \
            --model_name $3 \
            --log_dir  $4 \
            --lr  $5    \
            --batch_size $6 \
            --max_epochs $7 \
            --layer_num $8 \
            --max_length 300 \
            --template_id $9 \
            --full_mode
}






GPU=0
LOG_DIR="logs/ablation_4090"


ALL_LAYER=`echo " 0 1 2 3 4 5 6 7 8 9 10 11 12"`  #COMMENT
ALL_TID=`echo "0 1 2 3"`  #TID
BATCH_SIZE=30
LR=5e-5
# ALLDS=`echo "COMMENT SATD SMELL CODE"`  #COMMENT


# TID  CODE:1   COMMENT:0  SATD:1  SMELL:3
#ALLDS=`echo "CODE"`  #COMMENT

for LAYER in $ALL_LAYER
do 

     # echo "nothing"
      run_codeprompt COMMENT 885 self $LOG_DIR  $LR  $BATCH_SIZE 20 $LAYER 0
      run_codeprompt SATD 885 self $LOG_DIR  $LR  $BATCH_SIZE 20 $LAYER 1
      run_codeprompt SMELL 885 self $LOG_DIR  $LR  $BATCH_SIZE 20 $LAYER 3
      run_codeprompt CODE 885 self $LOG_DIR  $LR  $BATCH_SIZE 20 $LAYER 1
      
done


run_codeprompt COMMENT 885 ffn $LOG_DIR  $LR  $BATCH_SIZE 20 0 0
run_codeprompt SATD 885 ffn $LOG_DIR  $LR  $BATCH_SIZE 20 0 1
run_codeprompt SMELL 885 ffn $LOG_DIR  $LR  $BATCH_SIZE 20 0 3
run_codeprompt CODE 885 ffn $LOG_DIR  $LR  $BATCH_SIZE 20 0 1
