import os
import random

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.3'

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append('/data/mxl/rlhf/LaMP/PLoRA')

import json
import jsonlines
import shutil
import argparse
from datetime import datetime

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModel
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback

from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor

from datasets import DatasetDict
from datasets import Dataset as HFDataset

from data.datasets import (
    get_all_labels,
    GeneralSeq2SeqDataset,
    create_preprocessor,
    convert_to_hf_dataset
)

from peft import (
    # get_peft_model, MultitaskPromptTuningConfig,
    # TaskType, MultitaskPromptTuningInit,
    LoraConfig,
    PrefixTuningConfig,
)
from peft.tuners import lora
from ppeft import PeftModel
from torch.nn import Linear
from peft.peft_model import PrefixEncoder, PromptEncoder

from ppeft.plora import PLoraConfig
from ppeft.pplug import PPrefixTuningConfig, PPromptEncoderConfig, PPromptTuningConfig
# from plms.llama import LlamaForCausalLM

from transformers.models.llama import LlamaForCausalLM
# from transformers.models.t5 import T5ForConditionalGeneration
# from ppeft.mapping import get_peft_model

from jinja2 import Template

from data_loader import (
    load_dataset4plora_id,
    load_dataset4plora_profile,
    load_dataset4plora_moe,
    load_dataset4pplug_history,
    #
    load_dataset4short_cut,
    load_dataset4da,
)


def main():
    from args import process_args, make_exp_dirs
    model_args, data_args, training_args = process_args(make_dir=False)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    task = data_args.dataset_name
    print(f'process dataset {task}')

    if task in ['LaMP_1', 'LaMP_2', 'LaMP_t_1', 'LaMP_t_2', ]:
        labels = get_all_labels(task)
        best_metric = "accuracy"
        greater_is_better = True
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
    elif task in ['LaMP_3', 'LaMP_t_3', 'IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:  # class 5 10 5
        labels = get_all_labels(task)
        best_metric = "mae"
        greater_is_better = False
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels)
    else:
        best_metric = "rouge-1"
        greater_is_better = True
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    if model_args.personal_type in ['profile']:  # ['profile', 'uid']:
        if model_args.peft_type == 'MOE':
            my_dataset = load_dataset4plora_moe(model_args, data_args, training_args, tokenizer)
        else:
            my_dataset = load_dataset4plora_profile(model_args, data_args, training_args, tokenizer)
            # my_dataset = load_dataset4short_cut(model_args, data_args, training_args, tokenizer)
            # my_dataset = load_dataset4da(model_args, data_args, training_args, tokenizer)

    elif model_args.personal_type in ['history']:
        my_dataset = load_dataset4pplug_history(model_args, data_args, training_args, tokenizer)

    elif model_args.personal_type in ['uid']:
        my_dataset = load_dataset4plora_id(model_args, data_args, training_args, tokenizer)

    else:
        print(f'No implementation for {model_args.personal_type}')

    if model_args.peft_type == 'LORA':
        from transformers.models.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,  # 常数，类似于学习率
            lora_dropout=model_args.lora_dropout,
            # target_modules=['q', 'v'],
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
            bias="none",
            task_type='SEQ_2_SEQ_LM',  # "CAUSAL_LM",
        )
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)

    elif model_args.peft_type == 'ADALORA':  # "ADALORA": AdaLoraModel,
        from transformers.models.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        from peft import AdaLoraModel, AdaLoraConfig
        peft_config = AdaLoraConfig(
            peft_type="ADALORA",
            task_type="SEQ_2_SEQ_LM",
            init_r=model_args.lora_r,  # 32
            lora_alpha=model_args.lora_alpha,  # 32  16
            # target_modules=["q", "v"],
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
            lora_dropout=0.01,
        )
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)

    elif model_args.peft_type in ['PREFIX_TUNING', 'P_TUNINGv2']:
        # PREFIX_TUNING  暂时有问题
        from peft import get_peft_model
        from peft.tuners import PrefixEncoder
        from transformers.models.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        peft_config = PrefixTuningConfig(
            inference_mode=False,
            peft_type="PREFIX_TUNING",
            prefix_projection=True if model_args.peft_type == 'PREFIX_TUNING' else False,
            task_type="SEQ_2_SEQ_LM",
            encoder_hidden_size=128,  # 编码器 隐藏层 768/128
            num_virtual_tokens=20,  # embedding size
        )
        model = get_peft_model(model, peft_config)
        """
        检查模型梯度
        [(n,p.grad) fro n,p in model.named_parameters() if p.requires_grad]
        """

    elif model_args.peft_type in ['PLORA', 'MOE']:
        """
        PLRA  p=uid/profile
        """
        from plms import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        peft_config = PLoraConfig(
            user_feature_dim=model_args.user_feature_dim,

            personal_type=model_args.personal_type,
            inject_method=model_args.inject_method,

            user_size=model_args.user_size,
            expert_num=model_args.expert_num,
            target_layers=model_args.lora_target_layers.split(","),
            #
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type='SEQ_2_SEQ_LM',  # "CAUSAL_LM",
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
        )
        from ppeft.mapping import get_peft_model
        model = get_peft_model(model, peft_config)

        for n, p in model.named_parameters():
            if 'lora' in n:
                p.requires_grad = True

    elif model_args.peft_type in ['PPLUG', "PPROMPT_TUNING", "PPREFIX_TUNING", "PP_TUNING"]:
        """
        input_embedding based
        p-tuning(2103)  Embedding(text repeat init token)
        prompt_tuning(2104)  Embedding + Encoder(LSTM/MLP)
    
        past_key_values based
        prefix_tuning(2101)  Embedding + Encoder
        P-tuning(2110) v2 Embedding
        """
        from plms import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

        peft_config = PPrefixTuningConfig(
            inference_mode=False,
            peft_type="PPREFIX_TUNING",
            prefix_projection=True if model_args.peft_type == 'PPREFIX_TUNING' else False,
            task_type="SEQ_2_SEQ_LM",
            token_dim=768,
            # num_transformer_submodules=1,
            num_attention_heads=model.config.num_attention_heads,  # 12
            num_layers=model.config.num_hidden_layers,
            encoder_hidden_size=model.config.hidden_size,  # 768
            num_virtual_tokens=20 + (model_args.usr_size if model_args.personal_type == 'uid' else 0),  # embedding size
            # uid 情况用于设置 可学习 uid embedding
            personal_type=model_args.personal_type,
            user_size=5,  # model_args.user_size,  # 划分 nv [task_token, user_his_token]
        )
        from ppeft.mapping import get_peft_model
        model = get_peft_model(model, peft_config)

    else:
        print(f'No implementation for peft_type {model_args.peft_type}')

    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=training_args.max_seq_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{training_args.output_dir}',  # /cpts',
        logging_dir=f'{training_args.output_dir}/logs',
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",  # 评估
        eval_accumulation_steps=1,

        logging_steps=training_args.logging_steps,  # 日志步长
        load_best_model_at_end=True,  # 训练结束加载最佳模型  PREFIX_TUNING 加载 需要在 inference_mode, 训练和加载模型冲突
        # metric_for_best_model=best_metric,  # 评估指标
        # greater_is_better=greater_is_better,  # 指标越大越好

        # save_strategy="no",  # 保存检查点  epoch / no
        save_strategy="epoch",  # 保存检查点  epoch / no
        metric_for_best_model='eval_loss',  # 评估指标
        greater_is_better=False,

        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        num_train_epochs=training_args.num_train_epochs,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        predict_with_generate=True,
        save_safetensors=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=my_dataset['train'],
        eval_dataset=my_dataset['dev'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 输入为logits
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # need save check point

    )

    print('start train')

    make_exp_dirs(model_args, data_args, training_args)

    # check model [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    trainer.train()

    results = trainer.evaluate(my_dataset['test'])  # TODO:修改，在不训练时加载CPT
    print(results)

    with open(f'{training_args.output_dir}/results/dev_output.json', 'w') as file:
        json.dump(results, file, indent=4)

    trainer.save_model(f'{training_args.output_dir}/best_cpt', )  # 保存最佳模型
    shutil.rmtree(f'{training_args.output_dir}/cpts')  # 递归删除


def run_disturb():
    from args import process_args, make_exp_dirs
    model_args, data_args, training_args = process_args(make_dir=False)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    task = data_args.dataset_name
    print(f'process dataset {task}')

    if task in ['LaMP_1', 'LaMP_2', ]:
        labels = get_all_labels(task)
        best_metric = "accuracy"
        greater_is_better = True
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
    elif task in ['LaMP_3', 'IMDB-B', 'YELP-B', "GDRD-B", 'PPR-B']:  # class 5 10 5
        labels = get_all_labels(task)
        best_metric = "mae"
        greater_is_better = False
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels)
    else:
        best_metric = "rouge-1"
        greater_is_better = True
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    for dsr in [0.1 * i for i in range(10, 11)]:

        data_args.disturb_rate = dsr
        print('disturb rate', dsr)

        if model_args.personal_type in ['profile']:  # ['profile', 'uid']:
            # my_dataset = load_dataset4short_cut(model_args, data_args, training_args, tokenizer)
            my_dataset = load_dataset4da(model_args, data_args, training_args, tokenizer)

        if model_args.peft_type == 'LORA':
            from transformers.models.t5 import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,  # 常数，类似于学习率
                lora_dropout=model_args.lora_dropout,
                # target_modules=['q', 'v'],
                target_modules=model_args.lora_target_modules.split(",")
                if model_args.lora_target_modules != "all-linear"
                else model_args.lora_target_modules,
                bias="none",
                task_type='SEQ_2_SEQ_LM',  # "CAUSAL_LM",
            )
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)

        elif model_args.peft_type in ['PLORA', 'MOE']:
            """
            PLRA  p=uid/profile
            """
            from plms import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            peft_config = PLoraConfig(
                user_feature_dim=model_args.user_feature_dim,

                personal_type=model_args.personal_type,
                inject_method=model_args.inject_method,

                user_size=model_args.user_size,
                expert_num=model_args.expert_num,
                target_layers=model_args.lora_target_layers.split(","),
                #
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                r=model_args.lora_r,
                bias="none",
                task_type='SEQ_2_SEQ_LM',  # "CAUSAL_LM",
                target_modules=model_args.lora_target_modules.split(",")
                if model_args.lora_target_modules != "all-linear"
                else model_args.lora_target_modules,
            )
            from ppeft.mapping import get_peft_model
            model = get_peft_model(model, peft_config)

            for n, p in model.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True

        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=training_args.max_seq_length)

        s2s_training_args = Seq2SeqTrainingArguments(
            output_dir=f'{training_args.output_dir}',  # /cpts',
            logging_dir=f'{training_args.output_dir}/logs',
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",  # 评估
            eval_accumulation_steps=1,

            save_strategy="no",  # 保存检查点  epoch / no
            logging_steps=training_args.logging_steps,  # 日志步长
            # load_best_model_at_end=True,  # 训练结束加载最佳模型  PREFIX_TUNING 加载 需要在 inference_mode, 训练和加载模型冲突
            metric_for_best_model=best_metric,  # 评估指标
            greater_is_better=greater_is_better,  # 指标越大越好

            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_train_epochs=training_args.num_train_epochs,
            lr_scheduler_type=training_args.lr_scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            predict_with_generate=True,
            save_safetensors=False
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=s2s_training_args,
            data_collator=collator,
            train_dataset=my_dataset['train'],
            eval_dataset=my_dataset['dev'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,  # 输入为logits
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

        )
        print('start train')
        make_exp_dirs(model_args, data_args, training_args)
        trainer.train()
        # results = trainer.evaluate(my_dataset['test'])  # TODO:修改，在不训练时加载CPT
        # print(results)
        # with open(f'{training_args.output_dir}/results/dev_output.json', 'w') as file:
        #     json.dump(results, file, indent=4)
        # trainer.save_model(f'{training_args.output_dir}/best_cpt', )  # 保存最佳模型


if __name__ == "__main__":
    main()
    # run_disturb()

    """
    --seed
    1
    --output_dir
    output
    --model_name_or_path
    /data/hf/google/flan-t5-base
    --personal_type
    profile
    --peft_type
    PLORA
    --dataset_name
    LaMP_3
    --dataset_text_field
    text
    --train_method
    SFT
    --lora_r
    32
    --lora_alpha
    16
    --lora_dropout
    0.1
    --lora_target_modules
    q_proj,k_proj,v_proj
    --add_special_tokens
    False
    --append_concat_token
    False
    --max_seq_length
    2048
    --num_train_epochs
    2
    --logging_steps
    5
    --log_level
    info
    --logging_strategy
    steps
    --eval_strategy
    epoch
    --save_strategy
    steps
    --save_steps
    50
    --learning_rate
    1e-4
    --lr_scheduler_type
    cosine
    --weight_decay
    1e-4
    --warmup_ratio
    0.1
    --max_grad_norm
    1.0
    --per_device_train_batch_size
    8
    --per_device_eval_batch_size
    8
    --gradient_accumulation_steps
    8
    --gradient_checkpointing
    True
    --remove_unused_columns
    True
    """
