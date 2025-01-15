
# nohup bash run_ppeft.sh >run_ppeft.out 2>&1 &
# nohup bash run_llama.sh >run_llama_32_lora.out 2>&1 &
# nohup bash run_llama.sh >run_llama_8_lora_wpr2.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

#personal_types=('uid' 'profile' 'history')
#personal_types=('history')
personal_types=('profile')

#peft_types=('PLORA' 'LORA' 'PPLUG')
#peft_types=('PLORA' 'LORA' 'MOE')
peft_types=('PLORA')
#peft_types=('MOE')
#peft_types=('PPREFIX_TUNING')
#peft_types=('PREFIX_TUNING' 'P_TUNINGv2')

#model=/data/hf/meta-llama/Meta-Llama-3-8B-Instruct
model=/data/hf/meta-llama/Llama-3.2-1B-Instruct

#for task_id in 1 3 4 5 7

for task_id in 3
do
  for personal_type in "${personal_types[@]}"
  do
    task=LaMP_${task_id}

    num_peft_types=${#peft_types[@]}
    for ((i=0; i<$num_peft_types; i++))
    do
       pt=${peft_types[$i]}
#       gid=$((i%2))
#      pt='LORA'
      gid=0
#      pt='PLORA'
#      gid=0
#--lora_target_modules all-linear \


       echo run with peft method ${pt} ${personal_type} gid ${gid}

       CUDA_VISIBLE_DEVICES=${gid} python train_llama.py \
        --use_4bit_quantization True \
        --use_nested_quant True \
        --bnb_4bit_compute_dtype bfloat16 \
        --fp16 True \
        --expert_num 4 \
        --user_size 5 \
        --seed 1 \
        --output_dir output \
        --model_name_or_path ${model} \
        --personal_type ${personal_type} \
        --lora_target_modules 'q_proj,v_proj' \
        --prompt 'prompt,profile,history,query' \
        --num_retrieved 2 \
        --peft_type ${pt} \
        --dataset_name ${task} \
        --dataset_text_field text \
        --train_method SFT \
        --lora_r 32 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --add_special_tokens False \
        --append_concat_token False \
        --max_seq_length 1024 \
        --num_train_epochs 3 \
        --logging_steps 200 \
        --log_level info \
        --logging_strategy steps \
        --eval_strategy steps \
        --save_steps 1000 \
        --save_strategy steps \
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
done