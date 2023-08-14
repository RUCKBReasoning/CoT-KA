export MODEL_NAME=deberta-v3-large
export MODEL_TYPE=deberta-v3-large
export DATA_DIR=./data/arithmetic_reasoning/aqua/singlecot-zeroshot

set -x
GPUS=$1

# If you want to run a single experiment, use: for i in 0;
# This indicates that seed 0 is selected for a single experiment.

# If you want to run multiple experiments and average the results, use: for i in 0 1 2 3 4 5 6 7 8 9;
# This means the results are averaged over 10 random runs (as presented in our paper).
# Please note that running this experiment may take a considerable amount of time.

# For more details, refer to Chapter 5.2 and Appendix A.2 in our paper.


for i in 0; do
# for i in 0 1 2 3 4 5 6 7 8 9; do
  export TASK_NAME_=arithmetic_reasoning/aqua_singlecot-zeroshot_${i}_lr1e-5
  SEED=$(($i*10))
  echo $TASK_NAME_
  CUDA_VISIBLE_DEVICES=$GPUS python run_classify.py \
    --train_file $DATA_DIR/train_${SEED}.json \
    --validation_file $DATA_DIR/dev_${SEED}.json \
    --test_file $DATA_DIR/test_${SEED}.json \
    --task aqua \
    --model_name_or_path $MODEL_NAME \
    --max_seq_length 512 \
    --output_dir ./checkpoints/$TASK_NAME_/$MODEL_TYPE \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --warmup_steps 0 \
    --num_train_epochs 10 \
    --logging_steps 20 \
    --fp16 \
    --overwrite_output_dir \
    --max_steps 2000 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --save_total_limit 2 \
    --pad_to_max_length False \
    --seed $SEED
done
