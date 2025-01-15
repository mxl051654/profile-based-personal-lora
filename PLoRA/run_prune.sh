# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0


##personal_types=('uid' 'profile' 'history')
##peft_types=('PLORA' 'LORA' 'PPLUG')
##peft_types=('PLORA' 'LORA' 'MOE')
##peft_types=('PPREFIX_TUNING')
##peft_types=('PREFIX_TUNING' 'P_TUNINGv2')
##model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base

#########################################################################################################
# nohup bash run_prune.sh >run_dif_block.out 2>&1 &

#target_blocks=('e0,e1,e2' 'e3,e4,e5' 'e6,e7,e8' 'e9,e10,e11' 'd0,d1,d2' 'd3,d4,d5' 'd6,d7,d8' 'd9,d10,d11' )
#for tar in "${target_blocks[@]}"
#do
#  for task_id in 3 4 5 7
#  do
#    personal_type='profile'
#    task=LaMP_${task_id}
#    pt='PLORA'
#    echo run with peft method ${pt} ${personal_type}
#    CUDA_VISIBLE_DEVICES=0 python train_t5.py \
#      --suffix ${tar}_ \
#      --expert_num 4 \
#      --user_size 5 \
#      --seed 1 \
#      --output_dir output \
#      --model_name_or_path ${model} \
#      --personal_type ${personal_type} \
#      --peft_type ${pt} \
#      --dataset_name ${task} \
#      --dataset_text_field text \
#      --train_method SFT \
#      --lora_target_layers ${tar} \
#      --lora_target_modules q,k,v \
#      --lora_r 32 \
#      --lora_alpha 16 \
#      --lora_dropout 0.1 \
#      --add_special_tokens False \
#      --append_concat_token False \
#      --max_seq_length 512 \
#      --num_train_epochs 5 \
#      --logging_steps 200 \
#      --log_level info \
#      --logging_strategy steps \
#      --eval_strategy epoch \
#      --save_strategy steps \
#      --save_steps 1000 \
#      --learning_rate 5e-4 \
#      --lr_scheduler_type cosine \
#      --weight_decay 5e-3 \
#      --warmup_ratio 0.05 \
#      --max_grad_norm 1.0 \
#      --per_device_train_batch_size 8 \
#      --per_device_eval_batch_size 8 \
#      --gradient_accumulation_steps 2 \
#      --gradient_checkpointing True \
#      --remove_unused_columns True
#  done
#done


##################   module     #######################
#
## nohup bash run_prune.sh >run_dif_module.out 2>&1 &
#
#target_modules=('q' 'k' 'v' 'o' 'wi_0' 'wi_1' 'wo')
#for tar in "${target_modules[@]}"
#do
#  for task_id in 3 4 5 7
#  do
#    personal_type='profile'
#    task=LaMP_${task_id}
#    pt='PLORA'
#    echo run with peft method ${pt} ${personal_type}
#    CUDA_VISIBLE_DEVICES=1 python train_t5.py \
#      --suffix ${tar}_ \
#      --expert_num 4 \
#      --user_size 5 \
#      --seed 1 \
#      --output_dir output \
#      --model_name_or_path ${model} \
#      --personal_type ${personal_type} \
#      --peft_type ${pt} \
#      --dataset_name ${task} \
#      --dataset_text_field text \
#      --train_method SFT \
#      --lora_target_modules ${tar} \
#      --lora_r 32 \
#      --lora_alpha 16 \
#      --lora_dropout 0.1 \
#      --add_special_tokens False \
#      --append_concat_token False \
#      --max_seq_length 512 \
#      --num_train_epochs 5 \
#      --logging_steps 200 \
#      --log_level info \
#      --logging_strategy steps \
#      --eval_strategy epoch \
#      --save_strategy steps \
#      --save_steps 500 \
#      --learning_rate 5e-4 \
#      --lr_scheduler_type cosine \
#      --weight_decay 5e-3 \
#      --warmup_ratio 0.05 \
#      --max_grad_norm 1.0 \
#      --per_device_train_batch_size 8 \
#      --per_device_eval_batch_size 8 \
#      --gradient_accumulation_steps 2 \
#      --gradient_checkpointing True \
#      --remove_unused_columns True
#  done
#done


##################   module_mask_percent     #######################

# nohup bash run_prune.sh >run_plora_w_mask_24680.out 2>&1 &

mask_percent=(40 60 80)
for mp in "${mask_percent[@]}"
do
  for task_id in 1 3 4 5 7
  do
    personal_type='profile'
    task=LaMP_${task_id}
    pt='PLORA'
    echo run with peft method ${mp} ${pt} ${personal_type}
    CUDA_VISIBLE_DEVICES=1 python train_t5_w_mask.py \
      --suffix mask${mp}_ \
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
      --module_mask_percent ${mp} \
      --lora_target_modules all-linear \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.1 \
      --add_special_tokens False \
      --append_concat_token False \
      --max_seq_length 512 \
      --num_train_epochs 5 \
      --logging_steps 1000 \
      --log_level info \
      --logging_strategy steps \
      --eval_strategy epoch \
      --learning_rate 5e-4 \
      --lr_scheduler_type cosine \
      --weight_decay 5e-3 \
      --warmup_ratio 0.05 \
      --max_grad_norm 1.0 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 2 \
      --gradient_checkpointing True \
      --remove_unused_columns True
  done
done