function run_codeprompt_infer_att
{

# $1 dataset, $2 support_per_class, $3 per class, $4 shot ,$5 groupname  per class
#//microsoft/codebert-base, //microsoft/codebert-base-mlm, #microsoft/codebert-base-mlm  \
  CUDA_VISIBLE_DEVICES=$GPU  python codeprompt_infer.py  \
            --plm_name  microsoft/codebert-base-mlm    \
            --dataset $1  \
            --verbalizer none \
            --seed    $2 \
            --model_name $3 \
            --log_dir  $4 \
            --layer_num 2 \
            --att_visual \
            --att_visual_id $5 \
            --time_test \
            --max_length 256
}






GPU=0
LOG_DIR="logs/att_visual"
ALL_SAMPLE_ID=`echo "144"` 
BATCH_SIZE=30
ALLDS=`echo "CODE"`  #COMMENT


# TID  CODE:1   COMMENT:0  SATD:1  SMELL:3
#ALLDS=`echo "CODE"`  #COMMENT
LR=0
for SID in $ALL_SAMPLE_ID
do
  for DS in $ALLDS
  do 

      # echo "nothing"
        run_codeprompt_infer_att $DS 885 self $LOG_DIR $SID

        
  done
done
