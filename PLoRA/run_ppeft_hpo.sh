
# nohup bash run_ppeft_hpo.sh >run_yelp_hpo.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

#personal_types=('uid' 'profile' 'history')
#personal_types=('history')
personal_types=('profile')

#peft_types=('PLORA' 'LORA' 'PPLUG')
#peft_types=('PLORA' 'LORA' 'MOE')
#peft_types=('PLORA')
peft_types=('PLORA')
#model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base

inject_method='add'
ranks=(8)  # 8，16 32
#tasks=('IMDB-B')
tasks=('YELP-B')
for r in "${ranks[@]}"
do
  for task in "${tasks[@]}"
  do
    for personal_type in "${personal_types[@]}"
    do
      for pt in "${peft_types[@]}"
      do
         echo run ${task} ${personal_type} ${pt} ${r}
         CUDA_VISIBLE_DEVICES=1 python train_t5.py \
          --suffix ${r}_ \
          --prompt 'prompt,history,query' \
          --expert_num 4 \
          --user_size 5 \
          --seed 1 \
          --num_retrieved
          --output_dir output \
          --model_name_or_path ${model} \
          --personal_type ${personal_type} \
          --inject_method ${inject_method} \
          --peft_type ${pt} \
          --dataset_name ${task} \
          --dataset_text_field text \
          --train_method SFT \
          --lora_r ${r} \
          --lora_alpha 16 \
          --lora_dropout 0.1 \
          --lora_target_modules all-linear \
          --add_special_tokens False \
          --append_concat_token False \
          --max_seq_length 512 \
          --num_train_epochs 7 \
          --logging_steps 500 \
          --log_level info \
          --logging_strategy steps \
          --eval_strategy epoch \
          --save_strategy steps \
          --save_steps 100 \
          --learning_rate 5e-4 \
          --lr_scheduler_type cosine \
          --weight_decay 5e-3 \
          --warmup_ratio 0.05 \
          --max_grad_norm 1.0 \
          --per_device_train_batch_size 8 \
          --per_device_eval_batch_size 64 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing True \
          --remove_unused_columns True
      done
    done
  done
done