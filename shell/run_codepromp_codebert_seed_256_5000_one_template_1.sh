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






GPU=1
LOG_DIR="logs/6000ada_seed_256_5000"


#ALL_LAYER=`echo "1 2 3 4 5 6 7 8 9 10 11 12"`  #COMMENT
#ALL_TID=`echo "0 1 2 3"`  #TID
BATCH_SIZE=30
LR=5e-5
SEED1=256
SEED2=5000
#ALLDS=`echo "COMMENT SATD SMELL CODE"`  #COMMENT
#ALLDS=`echo "CODE"`  #COMMENT
#for DS in $ALLDS
#do 
#  for TID  in $ALL_TID
#  do
      
      #run_codeprompt CODE $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
      run_codeprompt CODE $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1

      #run_codeprompt SMELL $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 3
      run_codeprompt SMELL $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 3


      #run_codeprompt COMMENT $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 0
      run_codeprompt COMMENT $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 0
    
      #run_codeprompt SATD $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
      run_codeprompt SATD $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1







#  done 

#done


