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
LOG_DIR="logs/3090_seed_884_929_smell_search"


#ALL_LAYER=`echo "1 2 3 4 5 6 7 8 9 10 11 12"`  #COMMENT
ALL_TID=`echo "2 3"`  #TID
BATCH_SIZE=30
LR=5e-5
SEED1=256
SEED2=5000
#ALLDS=`echo "COMMENT SATD SMELL CODE"`  #COMMENT
#ALLDS=`echo "CODE"`  #COMMENT
#for DS in $ALLDS
#do 
ALL_SEED=`echo "884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929"`
#ALLDS=`echo "COMMENT SATD SMELL CODE"`  #COMMENT
#ALLDS=`echo "CODE"`  #COMMENT
#for DS in $ALLDS
#do 

run_codeprompt SMELL 925 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
run_codeprompt SMELL 927 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
run_codeprompt SMELL 928 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
run_codeprompt SMELL 929 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1

for TID  in $ALL_TID
do
for SEED in $ALL_SEED
do     
      #run_codeprompt CODE $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
      #run_codeprompt CODE $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1

      
      
      
      
      
      run_codeprompt SMELL $SEED self $LOG_DIR  $LR  $BATCH_SIZE 20 2 $TID
      
      
      #run_codeprompt SMELL $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 3


      #run_codeprompt COMMENT $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 0
      #run_codeprompt COMMENT $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 0
    
      #run_codeprompt SATD $SEED1 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
      #run_codeprompt SATD $SEED2 self $LOG_DIR  $LR  $BATCH_SIZE 20 2 1
done








done 

#done
