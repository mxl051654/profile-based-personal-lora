
# nohup bash run_ppeft.sh >run_ppeft.out 2>&1 &
# nohup bash run_ppeft.sh >run_plora_32.out 2>&1 &
# nohup bash run_ppeft.sh >run_ppeft_t_ppq_phpq.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

#personal_types=('uid' 'profile' 'history')
#personal_types=('history' 'uid')
#personal_types=('profile')
#personal_types=('uid')
#peft_types=('PLORA')
#peft_types=('LORA')

personal_types=('history')
peft_types=('PPREFIX_TUNING')

#peft_types=('PLORA' 'LORA')
#peft_types=('PLORA' 'MOE')
#peft_types=('MOE')
#peft_types=('LORA' 'PREFIX_TUNING' 'P_TUNINGv2')
#peft_types=('PREFIX_TUNING' 'P_TUNINGv2')

#model=/data/hf/google/flan-t5-xxl
#model=/data/hf/google/flan-t5-xl
#model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base

inject_method='add'  # add cat hadamard
rank=8
alpha=8
lr=5e-4 # *5e-3 1e-4  2e-5 5e-3
device=0
epoch=5
modules=all-linear  # all-linear

#prompts=('prompt,profile,query' 'prompt,history,profile,query')
prompts=('prompt,history,query')
#prompts=('prompt,history,profile,query')

for prompt in "${prompts[@]}"
do

#tasks=('IMDB-B' 'YELP-B')
#for task in "${tasks[@]}"

for task_id in 3 4 5 7 1
do
  for personal_type in "${personal_types[@]}"
  do
#    task=LaMP_${task_id}
    task=LaMP_t_${task_id}
    for pt in "${peft_types[@]}"
    do
       echo run ${task} ${personal_type} ${pt} ${prompt}
       CUDA_VISIBLE_DEVICES=${device} python train_t5.py \
        --expert_num 4 \
        --user_size 10 \
        --suffix ${rank}_ \
        --prompt ${prompt} \
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
        --lora_alpha ${alpha} \
        --lora_dropout 0.1 \
        --lora_target_modules ${modules} \
        --add_special_tokens False \
        --append_concat_token False \
        --max_seq_length 512 \
        --num_train_epochs ${epoch} \
        --logging_steps 400 \
        --log_level info \
        --logging_strategy steps \
        --eval_strategy epoch \
        --save_strategy epoch \
        --learning_rate 5e-4 \
        --lr_scheduler_type cosine \
        --weight_decay ${lr} \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing True \
        --remove_unused_columns True
    done
  done
done
done