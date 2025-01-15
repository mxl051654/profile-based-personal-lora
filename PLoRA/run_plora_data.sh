
# nohup bash run_plora_data.sh >run_plora_data_h2.out 2>&1 &

# 添加路径到最前，并保留原始路径
export PYTHONPATH=/data/mxl/rlhf/LaMP/PLoRA:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=0

personal_types=('profile')
#peft_types=('PLORA')
peft_types=('PLORA')

#model=/data/hf/google/flan-t5-large
model=/data/hf/google/flan-t5-base
inject_method='add'
rank=8
alpha=8
lr=1e-4 # 1e-4  2e-5
device=1
epoch=30
modules=q,v,wi_0,wi_1,wo

#tasks=('IMDB-B' 'YELP-B' 'GDRD-B' 'PPR-B')
tasks=('GDRD-B' 'PPR-B')
#prompts=('prompt,profile,history,query' 'prompt,history,query' 'prompt,query')
prompts=('prompt,query')

for task in "${tasks[@]}"
do
  for prompt in "${prompts[@]}"
  do
    for personal_type in "${personal_types[@]}"
    do
      for pt in "${peft_types[@]}"
      do
         echo '!!!'
         echo run ${task} ${prompt} ${personal_type} ${pt}
         echo '!!!'
         CUDA_VISIBLE_DEVICES=${device} python train_t5.py \
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
          --logging_steps 200 \
          --log_level info \
          --logging_strategy steps \
          --eval_strategy epoch \
          --save_strategy steps \
          --save_steps 100 \
          --learning_rate ${lr} \
          --lr_scheduler_type cosine \
          --weight_decay 1e-4 \
          --warmup_ratio 0.05 \
          --max_grad_norm 1.0 \
          --per_device_train_batch_size 8 \
          --per_device_eval_batch_size 64 \
          --gradient_accumulation_steps 2 \
          --gradient_checkpointing True \
          --remove_unused_columns True
      done
    done
  done
done