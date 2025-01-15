
# nohup bash run_ablation.sh >run_ppl.out 2>&1 &
# nohup bash run_ablation.sh >run_ponly_wda.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH

personal_type='profile'
pt='PLORA'

#model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base
device=0

modules=all-linear
#inject_method='add'  # add cat hadamard ponly
inject_method='add'
rank=8
alpha=16
epoch=10
lr=5e-4
decay=5e-3
#prompt='prompt,profile,history,query'  # profile 挤占 history 空间
prompt='prompt,query'  # prompt only
prompt='prompt,profile,query'  # profile only
num_retr=2

# w/o

for task_id in 1 3 4 5 7
do
  task=LaMP_${task_id}
   echo run ${task} ${personal_type} ${pt} ${inject_method}
#   CUDA_VISIBLE_DEVICES=${device} python train_t5.py \
   CUDA_VISIBLE_DEVICES=${device} python train_t5_da.py \
    --suffix ${rank}_ \
    --prompt ${prompt} \
    --num_retrieved ${num_retr} \
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
    --max_seq_length 512 \
    --num_train_epochs ${epoch} \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 200 \
    --eval_strategy epoch \
    --save_strategy steps \
    --save_steps 100 \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --weight_decay ${decay} \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --remove_unused_columns True
done