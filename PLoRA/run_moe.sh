
# nohup bash run_moe.sh >run_moe.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

#personal_types=('uid' 'profile' 'history')
personal_types=('profile')
peft_types=('MOE')

#model=/data/hf/meta-llama/Meta-Llama-3-8B-Instruct
#model=/data/hf/meta-llama/Llama-3.2-1B-Instruct
model=/data/hf/google/flan-t5-large
#model=/data/hf/google/flan-t5-base

for task_id in 4
do
  for personal_type in "${personal_types[@]}"
  do
    task=LaMP_${task_id}
    for pt in "${peft_types[@]}"
    do
       echo run with peft method ${pt} ${personal_type}
       CUDA_VISIBLE_DEVICES=1 python train_t5.py \
        --expert_num 4 \
        --user_size 5 \
        --seed 1 \
        --output_dir output \
        --model_name_or_path ${model} \
        --personal_type ${personal_type} \
        --peft_type ${pt} \
        --dataset_name ${task} \
        --dataset_text_field text \
        --train_method SFT \
        --lora_r 32 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_target_modules q,k,v \
        --add_special_tokens False \
        --append_concat_token False \
        --max_seq_length 512 \
        --num_train_epochs 10 \
        --logging_steps 50 \
        --log_level info \
        --logging_strategy steps \
        --eval_strategy epoch \
        --save_strategy steps \
        --save_steps 100 \
        --learning_rate 5e-4 \
        --lr_scheduler_type cosine \
        --weight_decay 1e-2 \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing True \
        --remove_unused_columns True
    done
  done
done