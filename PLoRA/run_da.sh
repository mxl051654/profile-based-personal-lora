
# nohup bash run_da.sh >run_da_17p_s2.out 2>&1 &
# nohup bash run_da.sh >run_da_t5large_s2.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH

personal_types=('profile')
peft_types=('PLORA')

#model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base

inject_method='add'  # add cat hadamard
rank=8
lr=5e-3
device=0

#tasks=('LaMP_1' 'LaMP_7' 'IMDB-B' 'YELP-B' 'GDRD-B' 'PPR-B')

for task_id in 1 4 7
do
  for personal_type in "${personal_types[@]}"
  do
    task=LaMP_${task_id}
    for pt in "${peft_types[@]}"
    do
       echo '!!!!'
       echo run ${task} ${personal_type} ${pt} ${model}
       echo '!!!!'
       CUDA_VISIBLE_DEVICES=${device} python train_t5_da.py \
        --suffix ${rank}_ \
        --prompt 'prompt,history,query' \
        --num_retrieved 2 \
        --seed 1 \
        --output_dir output \
        --model_name_or_path ${model} \
        --personal_type ${personal_type} \
        --inject_method ${inject_method} \
        --peft_type ${pt} \
        --dataset_name ${task} \
        --dataset_text_field text \
        --train_method SFT \
        --lora_r ${rank} \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_target_modules all-linear \
        --add_special_tokens False \
        --append_concat_token False \
        --max_seq_length 512 \
        --num_train_epochs 5 \
        --logging_steps 200 \
        --log_level info \
        --logging_strategy steps \
        --eval_strategy epoch \
        --save_strategy steps \
        --save_steps 500 \
        --learning_rate ${lr} \
        --lr_scheduler_type cosine \
        --weight_decay 5e-3 \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing True \
        --remove_unused_columns True
    done
  done
done