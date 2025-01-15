import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.3'

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append('/data/mxl/rlhf/LaMP/PLoRA')

from math import sqrt
import json
import jsonlines
import shutil
import argparse
from datetime import datetime

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModel, \
    T5ForConditionalGeneration
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

from ppeft.plora import PLoraConfig
from ppeft.pplug import PPrefixTuningConfig, PPromptEncoderConfig, PPromptTuningConfig

from data_loader import (
    load_dataset4plora_id,
    load_dataset4plora_profile,
    load_dataset4plora_moe,
    load_dataset4pplug_history,
    load_dataset4shift_data,
    load_dataset4short_cut,

)
import matplotlib.pyplot as plt


def insert_probe(model, norms_dict):
    def probe_hook(name):
        """生成带模块名称的钩子函数"""

        def hook(module, input, output):
            # 使用模块的完整名称作为键
            if name not in norms_dict:
                norms_dict[name] = []
            # 记录输出的范数（L2 范数）
            norms_dict[name].append(torch.norm(output.detach()).item())

        return hook

    # 遍历模型，注册带有 `lora` 的模块的钩子
    for name, module in model.named_modules():
        # if 'lora_a' in name.lower() or 'lora_b' in name.lower or 'lora_p' in name.lower():
        if 'lora_b' in name.lower() and 'default' in name.lower():
            print(f"Registering hook for module: {name}")
            module.register_forward_hook(probe_hook(name))


def eval():
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
    elif task in ['LaMP_3']:
        labels = get_all_labels(task)
        best_metric = "mae"
        greater_is_better = False
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels)
    else:
        best_metric = "rouge-1"
        greater_is_better = True
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    # t5 all-linear  8
    # ad_path = {
    #     'LaMP_1': '/data/mxl/rlhf/LaMP/output/LaMP_1/SFT_PLORA/' + \
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241203-220946/best_cpt',
    #
    #     'LaMP_3': '/data/mxl/rlhf/LaMP/output/LaMP_3/SFT_PLORA/' + \
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241203-224939/best_cpt',
    #
    #     'LaMP_4': '/data/mxl/rlhf/LaMP/output/LaMP_4/SFT_PLORA/' + \
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241204-001203/best_cpt',
    #
    #     'LaMP_5': '/data/mxl/rlhf/LaMP/output/LaMP_5/SFT_PLORA/' + \
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241204-011838/best_cpt',
    #
    #     'LaMP_7': '/data/mxl/rlhf/LaMP/output/LaMP_7/SFT_PLORA/' + \
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241204-021931/best_cpt',
    # }[task]

    # TODO for test data augmentation robust
    # ad_path = {
    #     'LaMP_3': '/data/mxl/rlhf/LaMP/output/LaMP_3/SFT_PLORA/'
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_20241217-074755/best_cpt',
    #     'LaMP_4': '/data/mxl/rlhf/LaMP/output/LaMP_4/SFT_PLORA/'
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_20241218-095043/best_cpt',
    #     'LaMP_5': '/data/mxl/rlhf/LaMP/output/LaMP_5/SFT_PLORA/'
    #               'contriever_flan-t5-base_lr_0.0005_profile_8_20241218-035909/best_cpt',
    # }[task]

    ad_path = '/data/mxl/rlhf/LaMP/output/LaMP_3/SFT_LORA/' \
              'contriever_flan-t5-base_lr_5e-05_profile_20250102-164222/best_cpt'

    if model_args.peft_type == 'LORA':
        from transformers.models.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ad_path)

    elif model_args.peft_type in ['PLORA', 'MOE']:
        from plms import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            output_attentions=True)
        peft_config = PLoraConfig(
            user_feature_dim=model_args.user_feature_dim,
            personal_type=model_args.personal_type,
            user_size=model_args.user_size,
            expert_num=model_args.expert_num,
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
        # peft_config = PLoraConfig.from_pretrained(ad_path)
        from ppeft.mapping import get_peft_model
        model = get_peft_model(model, peft_config)

        cpt = torch.load(f'{ad_path}/pytorch_model.bin')
        model.load_state_dict(cpt)

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
        from ppeft import PeftModel
        model = PeftModel.from_pretrained(model, ad_path)

    else:
        print(f'No implementation for peft_type {model_args.peft_type}')

    # norms_dict = {}
    # insert_probe(model, norms_dict)

    my_dataset = None

    if model_args.personal_type in ['profile']:  # ['profile', 'uid']:
        # my_dataset = load_dataset4plora_profile(model_args, data_args, training_args, tokenizer)
        # my_dataset = load_dataset4shift_data(model_args, data_args, training_args, tokenizer)
        my_dataset = load_dataset4short_cut(model_args, data_args, training_args, tokenizer)

    elif model_args.personal_type in ['history']:
        my_dataset = load_dataset4pplug_history(model_args, data_args, training_args, tokenizer)

    elif model_args.personal_type in ['uid']:
        my_dataset = load_dataset4plora_id(model_args, data_args, training_args, tokenizer)

    else:
        print(f'No implementation for {model_args.personal_type}')

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=training_args.max_seq_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{training_args.output_dir}',  # /cpts',
        logging_dir=f'{training_args.output_dir}/logs',
        # do_train=True,
        do_eval=True,
        eval_strategy="epoch",  # 评估
        eval_accumulation_steps=1,

        save_strategy="epoch",  # 保存检查点
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
        args=training_args,
        data_collator=collator,
        train_dataset=my_dataset['train'],
        eval_dataset=my_dataset['dev'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 输入为logits
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    results = trainer.evaluate(my_dataset['dev'])  # TODO:修改，在不训练时加载CPT
    print('Result:', results)

    # TODO cal attention
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(
    #     my_dataset['dev'],
    #     batch_size=16,
    #     collate_fn=collator
    # )
    # saved_data = {
    #     "inputs": [],  # 保存每个batch的输入
    #     "outputs": [],  # 保存模型的输出 logits 或 loss
    #     "cross_attentions": [],  # 保存交叉注意力
    #     "encoder_attentions": [],  # 保存编码器注意力
    #     "decoder_attentions": []  # 保存解码器注意力
    # }
    #
    # for batch in dataloader:
    #     batch = batch.to(model.device)
    #     outputs = model(**batch)
    #     # cross_attentions = outputs.cross_attentions  # 12 (16 12 2 512)  layer batch head out in
    #     # encoder_attentions = outputs.encoder_attentions  # 12(16 12 512 512)
    #     # decoder_attentions = outputs.decoder_attentions  # 12 (16 12 2 2)
    #
    #     # 保存批次输入和模型输出
    #     saved_data["inputs"].append({k: v.detach().cpu() for k, v in batch.items()})
    #     saved_data["outputs"].append({
    #         "logits": outputs.logits.detach().cpu(),  # 保存 logits
    #         "loss": outputs.loss.detach().cpu() if outputs.loss is not None else None  # 保存 loss（如果存在）
    #     })
    #
    #     # 保存注意力权重
    #     if outputs.cross_attentions is not None:
    #         saved_data["cross_attentions"].append([
    #             layer.detach().cpu() for layer in outputs.cross_attentions  # 保存每一层的 cross attention
    #         ])
    #     if outputs.encoder_attentions is not None:
    #         saved_data["encoder_attentions"].append([
    #             layer.detach().cpu() for layer in outputs.encoder_attentions  # 保存每一层的 encoder attention
    #         ])
    #     if outputs.decoder_attentions is not None:
    #         saved_data["decoder_attentions"].append([
    #             layer.detach().cpu() for layer in outputs.decoder_attentions  # 保存每一层的 decoder attention
    #         ])
    #
    # # TODO save attention
    # # 将保存的数据写入文件
    # torch.save(saved_data, "model_inputs_outputs.pt")
    # print("模型输入和输出已保存到 'model_inputs_outputs.pt'")
    # loaded_data = torch.load("model_inputs_outputs.pt")
    # # 查看保存的信息
    # print("Saved Inputs:", loaded_data["inputs"])
    # print("Saved Outputs:", loaded_data["outputs"])
    # print("Cross Attentions:", len(loaded_data["cross_attentions"]))
    # print("Encoder Attentions:", len(loaded_data["encoder_attentions"]))
    # print("Decoder Attentions:", len(loaded_data["decoder_attentions"]))

    # module_k_v = {k: sum(v) / len(v) for k, v in norms_dict.items()}
    # with open(f"/data/mxl/rlhf/LaMP/output/{task}/SFT_PLORA/module_importance.json", 'w') as f:
    #     json.dump(module_k_v, f)


def att_ana():
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
    elif task in ['LaMP_3']:
        labels = get_all_labels(task)
        best_metric = "mae"
        greater_is_better = False
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels)
    else:
        best_metric = "rouge-1"
        greater_is_better = True
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    my_dataset = None
    if model_args.personal_type in ['profile']:  # ['profile', 'uid']:
        if model_args.peft_type == 'MOE':
            my_dataset = load_dataset4plora_moe(model_args, data_args, training_args, tokenizer)
        else:
            my_dataset = load_dataset4plora_profile(model_args, data_args, training_args, tokenizer)

    elif model_args.personal_type in ['history']:
        my_dataset = load_dataset4pplug_history(model_args, data_args, training_args, tokenizer)

    elif model_args.personal_type in ['uid']:
        my_dataset = load_dataset4plora_id(model_args, data_args, training_args, tokenizer)

    else:
        print(f'No implementation for {model_args.personal_type}')

    # t5 all-linear  8
    ad_path = {
        'LaMP_1': '/data/mxl/rlhf/LaMP/output/LaMP_1/SFT_PLORA/' + \
                  'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241203-220946/best_cpt',

        'LaMP_3': '/data/mxl/rlhf/LaMP/output/LaMP_3/SFT_PLORA/' + \
                  'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241203-224939/best_cpt',

        'LaMP_4': '/data/mxl/rlhf/LaMP/output/LaMP_4/SFT_PLORA/' + \
                  'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241204-001203/best_cpt',

        'LaMP_5': '/data/mxl/rlhf/LaMP/output/LaMP_5/SFT_PLORA/' + \
                  'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241204-011838/best_cpt',

        'LaMP_7': '/data/mxl/rlhf/LaMP/output/LaMP_7/SFT_PLORA/' + \
                  'contriever_flan-t5-base_lr_0.0005_profile_8_all-linear_20241204-021931/best_cpt',
    }[task]

    if model_args.peft_type == 'LORA':
        from transformers.models.t5 import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ad_path)

    elif model_args.peft_type in ['PLORA', 'MOE']:
        from plms import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            output_attentions=True)
        peft_config = PLoraConfig(
            user_feature_dim=model_args.user_feature_dim,
            personal_type=model_args.personal_type,
            user_size=model_args.user_size,
            expert_num=model_args.expert_num,
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
        # peft_config = PLoraConfig.from_pretrained(ad_path)
        from ppeft.mapping import get_peft_model
        model = get_peft_model(model, peft_config)

        cpt = torch.load(f'{ad_path}/pytorch_model.bin')
        model.load_state_dict(cpt)

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
        from ppeft import PeftModel
        model = PeftModel.from_pretrained(model, ad_path)

    else:
        print(f'No implementation for peft_type {model_args.peft_type}')

    # norms_dict = {}
    # insert_probe(model, norms_dict)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=training_args.max_seq_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{training_args.output_dir}',  # /cpts',
        logging_dir=f'{training_args.output_dir}/logs',
        # do_train=True,
        do_eval=True,
        eval_strategy="epoch",  # 评估
        eval_accumulation_steps=1,

        save_strategy="epoch",  # 保存检查点
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
        args=training_args,
        data_collator=collator,
        train_dataset=my_dataset['train'],
        eval_dataset=my_dataset['dev'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 输入为logits
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # results = trainer.evaluate(my_dataset['dev'])  # TODO:修改，在不训练时加载CPT
    # print('Result:', results)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        my_dataset['dev'],
        batch_size=16,
        collate_fn=collator
    )
    saved_data = {
        "inputs": [],  # 保存每个batch的输入
        "outputs": [],  # 保存模型的输出 logits 或 loss
        "cross_attentions": [],  # 保存交叉注意力
        "encoder_attentions": [],  # 保存编码器注意力
        "decoder_attentions": []  # 保存解码器注意力
    }
    from tqdm import tqdm
    for i, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(model.device)
        outputs = model(**batch)
        # cross_attentions = outputs.cross_attentions  # 12 (16 12 2 512)  layer batch head out in
        # encoder_attentions = outputs.encoder_attentions  # 12(16 12 512 512)
        # decoder_attentions = outputs.decoder_attentions  # 12 (16 12 2 2)

        # 保存批次输入和模型输出
        saved_data["inputs"].append({k: v.detach().cpu() for k, v in batch.items()})
        saved_data["outputs"].append({
            "logits": outputs.logits.detach().cpu(),  # 保存 logits
            "loss": outputs.loss.detach().cpu() if outputs.loss is not None else None  # 保存 loss（如果存在）
        })

        # 保存注意力权重
        if outputs.cross_attentions is not None:
            saved_data["cross_attentions"].append([
                layer.detach().cpu() for layer in outputs.cross_attentions  # 保存每一层的 cross attention
            ])
        if outputs.encoder_attentions is not None:
            saved_data["encoder_attentions"].append([
                layer.detach().cpu() for layer in outputs.encoder_attentions  # 保存每一层的 encoder attention
            ])
        if outputs.decoder_attentions is not None:
            saved_data["decoder_attentions"].append([
                layer.detach().cpu() for layer in outputs.decoder_attentions  # 保存每一层的 decoder attention
            ])
        if i == 10:
            break

    # 将保存的数据写入文件
    torch.save(saved_data, "model_inputs_outputs.pt")
    print("模型输入和输出已保存到 'model_inputs_outputs.pt'")

    loaded_data = torch.load("model_inputs_outputs.pt")

    # 查看保存的信息
    print("Saved Inputs:", loaded_data["inputs"])
    print("Saved Outputs:", loaded_data["outputs"])
    print("Cross Attentions:", len(loaded_data["cross_attentions"]))
    print("Encoder Attentions:", len(loaded_data["encoder_attentions"]))
    print("Decoder Attentions:", len(loaded_data["decoder_attentions"]))

    # 获取第1批次的第6层的交叉注意力，第1个头
    batch_idx = 0
    layer_idx = -1  # 第6层
    head_idx = 0  # 第1个注意力头

    # 提取特定的交叉注意力权重
    cross_attention = loaded_data["cross_attentions"][batch_idx][layer_idx]
    specific_attention = cross_attention[0, head_idx]  # 选择第1个样本和第1个头

    # 可视化特定层和头的注意力权重
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    sns.heatmap(specific_attention.numpy(), cmap="viridis", cbar=True)
    plt.title(f"Cross Attention (Layer {layer_idx + 1}, Head {head_idx + 1})")
    plt.xlabel("Source Tokens (Encoder)")
    plt.ylabel("Target Tokens (Decoder)")
    plt.show()


def view_attention(input_tokens, attention_weights):
    # 归一化注意力权重（范围[0, 1]）
    normalized_weights = (attention_weights - attention_weights.min()) / (
            attention_weights.max() - attention_weights.min())

    # 将tokens切分为字符序列，并映射权重到每个字符
    def split_tokens_to_characters(tokens, weights):
        char_list = []  # 保存所有字符
        char_weights = []  # 保存每个字符对应的权重
        for token, weight in zip(tokens, weights):
            for char in token:
                char_list.append(char)  # 添加字符
                char_weights.append(weight)  # 将token的权重映射到每个字符
            char_list.append(" ")  # 添加空格
            char_weights.append(0)  # 空格的权重设置为0
        return char_list, char_weights

    # 调用函数切分字符
    char_list, char_weights = split_tokens_to_characters(input_tokens, normalized_weights)

    # 按32个字符切分为多行
    def split_into_lines(chars, weights, max_chars=32):
        lines = []
        weight_lines = []
        current_line = []
        current_weights = []
        current_length = 0

        for char, weight in zip(chars, weights):
            if current_length + 1 > max_chars:  # 如果行的字符数超过最大值
                lines.append(current_line)
                weight_lines.append(current_weights)
                current_line = []
                current_weights = []
                current_length = 0
            current_line.append(char)
            current_weights.append(weight)
            current_length += 1
        if current_line:
            lines.append(current_line)
            weight_lines.append(current_weights)
        return lines, weight_lines

    # 切分字符为多行
    lines, weight_lines = split_into_lines(char_list, char_weights, max_chars=64)

    # 定义颜色映射函数（统一透明度为0.5）
    def get_color(weight):
        """
        根据注意力权重返回背景颜色，透明度为0.5：
        - 白色表示不重要 (权重接近0)
        - 蓝色加深表示重要 (权重越大，颜色越深)
        """
        if weight == 0:
            return (1, 1, 1, 0.5)  # 白色 + 透明度
        return (sqrt(1 - weight), sqrt(1 - weight), 1, 0.5)  # 蓝色 + 透明度

    # 绘制图像
    fig, ax = plt.subplots(figsize=(12, len(lines)))  # 高度根据行数动态调整

    # 遍历每一行
    y_offset = len(lines) - 1  # y轴从顶部开始
    x_start = 0  # x轴起点

    for line, weight_line in zip(lines, weight_lines):
        for char, weight in zip(line, weight_line):
            # 获取背景颜色
            color = get_color(weight)

            # 绘制带背景的字符
            ax.text(
                x_start, y_offset, s=char,  # 文本内容
                ha='center', va='center',  # 水平和垂直居中
                fontsize=13, color="black",  # 字体颜色
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1')  # 设置背景颜色，去掉边框
            )
            # 更新x轴位置（每个字符宽度为1）
            x_start += 1

        # 换行，重置x轴起点
        y_offset -= 0.3
        x_start = 0

    # 去掉多余的图像元素（如坐标轴）
    ax.axis("off")

    # 调整x轴和y轴范围
    ax.set_xlim(-1, 64)  # x轴范围（最大每行32字符）
    ax.set_ylim(-1, len(lines))  # y轴范围

    # 显示图片
    plt.title("Cross Attention Visualization (Character-level)", fontsize=16)
    plt.show()


def view():
    # saved_data = {
    #     "inputs": [],  # 保存每个batch的输入
    #     "outputs": [],  # 保存模型的输出 logits 或 loss
    #     "cross_attentions": [],  # 保存交叉注意力
    #     "encoder_attentions": [],  # 保存编码器注意力
    #     "decoder_attentions": []  # 保存解码器注意力
    # }
    # # 将保存的数据写入文件
    # torch.save(saved_data, "model_inputs_outputs.pt")
    # print("模型输入和输出已保存到 'model_inputs_outputs.pt'")
    from transformers import AutoTokenizer
    from args import process_args, make_exp_dirs
    model_args, data_args, training_args = process_args(make_dir=False)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    loaded_data = torch.load("model_inputs_outputs.pt")

    batch_idx = 0
    layer_idx = -1  # 第6层
    head_idx = 0  # 第1个注意力头

    # decode
    # tokenizer.decode(loaded_data['inputs'][0]['input_ids'][0])
    # tokenizer.batch_decode([[x] for x in loaded_data['inputs'][0]['input_ids'][0] if x!= tokenizer.pad_token and x!=tokenizer.eos_token])

    # 提取特定的交叉注意力权重
    cross_attention = loaded_data["cross_attentions"][batch_idx][layer_idx]
    specific_attention = cross_attention[0, head_idx]  # 选择第1个样本和第1个头

    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoTokenizer
    # 示例输入tokens（用于展示，通常从模型中获取）
    # input_tokens = ["Translate", "English", "to", "German", ":", "The", "book", "is", "on", "the", "table", "."]
    # attention_weights = specific_attention[0].detach().cpu().numpy()  # 获取注意力权重 (input_seq_len, )

    for i in range(8):
        sid = i + 2
        input_tokens = tokenizer.batch_decode([[x] for x in loaded_data['inputs'][batch_idx]['input_ids'][sid] if
                                               x != tokenizer.pad_token and x != tokenizer.eos_token])
        attention_weights = loaded_data["cross_attentions"][batch_idx][layer_idx][sid, head_idx][0,
                            :].detach().cpu().numpy()

        view_attention(input_tokens, attention_weights)

    # TODO 记录正确和错误样本，分析模式差异


if __name__ == "__main__":
    # main4peft()
    # eval()
    # att_ana()

    # view()
    eval()

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
