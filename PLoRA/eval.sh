
# nohup bash eval.sh >eval.out 2>&1 &

export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH

#personal_types=('uid' 'profile' 'history')
#personal_types=('history')
personal_types=('profile')

#peft_types=('PLORA' 'LORA' 'PPLUG')
#peft_types=('PLORA' 'LORA' 'MOE')
#peft_types=('MOE')
peft_types=('PLORA')
#peft_types=('PPREFIX_TUNING')
#peft_types=('PREFIX_TUNING' 'P_TUNINGv2')

#model=/data/hf/meta-llama/Meta-Llama-3-8B-Instruct
#model=/data/hf/meta-llama/Llama-3.2-1B-Instruct
#model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base

for task_id in 3 5 7 1
do
  for personal_type in "${personal_types[@]}"
  do
    task=LaMP_${task_id}
    for pt in "${peft_types[@]}"
    do
       echo run with peft method ${pt} ${personal_type}

       CUDA_VISIBLE_DEVICES=0 python eval.py \
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
        --num_train_epochs 5 \
        --logging_steps 50 \
        --log_level info \
        --logging_strategy steps \
        --eval_strategy epoch \
        --save_strategy steps \
        --save_steps 100 \
        --learning_rate 5e-3 \
        --lr_scheduler_type cosine \
        --weight_decay 1e-2 \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing True \
        --remove_unused_columns True
    done
  done
done
# --adapter_name_or_path ../output/LaMP_3/SFT_PLORA/contriever_Meta-Llama-3-8B-Instruct_lr_5e-05_uid20241106-154134 \

#CUDA_VISIBLE_DEVICES=1 python train_llama.py \
#  --output_dir output \
#  --model_name_or_path ${model} \
#  --dataset_name ${task} \
#  --dataset_text_field text \
#  --train_method SFT \
#  --personal_type uid \
#  --peft_type PLORA \
#  --lora_r 32 \
#  --lora_alpha 16 \
#  --lora_dropout 0.1 \
#  --lora_target_modules q_proj,k_proj,v_proj \
#  --add_special_tokens False \
#  --append_concat_token False \
#  --max_seq_len 1024 \
#  --num_train_epochs 1 \
#  --log_level info \
#  --logging_strategy steps \
#  --logging_steps 10 \
#  --eval_strategy epoch \
#  --save_strategy steps \
#  --save_steps 50 \
#  --learning_rate 5e-5 \
#  --lr_scheduler_type cosine \
#  --weight_decay 1e-4 \
#  --warmup_ratio 0.1 \
#  --max_grad_norm 1.0 \
#  --per_device_train_batch_size 8 \
#  --per_device_eval_batch_size 8 \
#  --gradient_accumulation_steps 8 \
#  --gradient_checkpointing True \
#  --remove_unused_columns True \
#  --use_4bit_quantization True \
#  --use_nested_quant True \
#  --bnb_4bit_compute_dtype bfloat16 \
#  --fp16 True \
#  --seed 1
