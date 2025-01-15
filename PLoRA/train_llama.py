"""
关于 clm 训练数据
https://huggingface.co/docs/trl/main/en/sft_trainer

数据处理
从sft_clm_mlm三种训练方式来看data_collator——【transformers源码阅读】
clm的特征，就是在训练的时候，只能看到左边的词。
mlm的特征，就说在训练的时候，可以看到两边的词。


DataCollatorForCompletionOnlyLM

# /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
#

"""

# TODO https://github.com/huggingface/peft/blob/main/examples/sft/train.py

import sys

sys.path.append('/data/mxl/rlhf/LaMP/PLoRA')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.3'

from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import jsonlines
from trl import DataCollatorForCompletionOnlyLM
from peft.tuners.lora import LoraConfig

from ppeft import PeftModel
from ppeft.plora import PLoraConfig
from plms import LlamaForCausalLM
from ppeft.mapping import get_peft_model
from lamp_data_loader import load_data4plora, load_data4lora
from metrics.my_metrics import load_metrics

from data_loader import (
    load_dataset4plora_id,
    load_dataset4plora_profile,
    load_dataset4plora_moe,
    load_dataset4pplug_history,
)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class VllmAGent:
    def __init__(self, model_name_or_path=None, adapter_name_or_path=None, beam_search=False):
        mem = 0.5
        temp = 0

        if adapter_name_or_path:
            self.lora_path = adapter_name_or_path[0]
            self.llm = LLM(
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                gpu_memory_utilization=mem,  # 最大显存预算, vllm kv缓存会占用显存
                max_model_len=4096,
                enable_lora=True,
                disable_log_stats=False,  # 显示进度
                disable_async_output_proc=True,
                max_lora_rank=64,  # default 16
            )
        else:
            self.lora_path = None
            self.llm = LLM(
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                gpu_memory_utilization=mem,  # 最大显存预算, vllm kv缓存会占用显存
                max_model_len=4096,
                disable_log_stats=False,  # 显示进度
                disable_async_output_proc=True,
                max_lora_rank=64,  # default 16
            )
        if not beam_search:
            self.sampling_params = SamplingParams(
                n=1,  # num_return_sequences=repeat_n,
                max_tokens=512,  # max_new_tokens=128,
                temperature=mem,  # 0 趋向于高频词，容易重复， 1< 原始 softmax, >1 区域均匀，随机
                top_k=20,  # 基于topK单词概率约束候选范围
                length_penalty=1,  # <0 鼓励长句子， >0 鼓励短句子
            )
        else:
            self.sampling_params = SamplingParams(
                use_beam_search=True,
                best_of=3,  # >1
                temperature=0,
                top_p=1,
                top_k=-1,
                max_tokens=512,
                length_penalty=0,  # <0 鼓励长句子， >0 鼓励短句子
                n=1,  # num_return_sequences=repeat_n,
            )

    def set_sampling_params(self, **kwargs):
        self.sampling_params = SamplingParams(**kwargs)

    def infer_vllm(self, inputs, instruction=None, chat=False, param=None):
        if param is not None:
            self.set_sampling_params(**param)

        assert isinstance(inputs, list)
        prompt_queries = []

        if instruction is None:
            instruction = 'you are a helpful assistant.'

        for x in inputs:
            if chat:
                message = [{"role": "system", "content": instruction},
                           {"role": "user", "content": x}]
            else:
                message = x
            prompt_queries.append(message)

        if chat:
            tokenizer = self.llm.get_tokenizer()
            model_config = self.llm.llm_engine.get_model_config()
            from vllm.entrypoints.chat_utils import apply_hf_chat_template, parse_chat_messages

            # https://hugging-face.cn/docs/transformers/chat_templating
            def apply_chat(x):
                conversation, _ = parse_chat_messages(x, model_config, tokenizer)
                prompt = apply_hf_chat_template(tokenizer, conversation=conversation,
                                                # 用于继续回答   {"role": "assistant", "content": '{"name": "'},
                                                # continue_final_message=True,
                                                # 推理需要 <|start_header_id|>assistant<|end_header_id|>
                                                add_generation_prompt=True,
                                                chat_template=None)  # 根据 tokenizer已经加载了模板，不需要自定义
                return prompt

            inputs = [apply_chat(x) for x in prompt_queries]

            outputs = self.llm.generate(
                inputs,
                self.sampling_params,
                use_tqdm=True,
                lora_request=LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None,
            )
        else:
            outputs = self.llm.generate(
                prompt_queries,
                self.sampling_params,
                use_tqdm=True,
                lora_request=LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None,
            )

        ret = []
        for output in outputs:
            ret.extend([x.text for x in output.outputs])
        return ret


def create_and_prepare_model(model_args, data_args, training_args):
    args = model_args

    bnb_config = None
    quant_storage_dtype = None

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)  # torch.bfloat16
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,  # nf4
            bnb_4bit_compute_dtype=compute_dtype,  # torch.bfloat16
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
    )
    # model = AutoModelForCausalLM.from_pretrained(
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        # trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype,
    )

    peft_config = None
    if args.peft_type == 'LORA':  # and not args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            peft_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=model_args.adapter_name_or_path)
        else:
            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                # q_proj, [k_proj], v_proj, gate_proj, down_proj, up_proj
                target_modules=args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules,
            )
    elif args.peft_type == 'PLORA':
        if model_args.adapter_name_or_path is not None:
            peft_config = PLoraConfig.from_pretrained(pretrained_model_name_or_path=model_args.adapter_name_or_path)
            # peft_config.personal_type = args.personal_type
        else:
            peft_config = PLoraConfig(
                # TODO for plora
                user_feature_dim=model_args.user_feature_dim,
                personal_type=model_args.personal_type,
                user_size=model_args.user_size,
                expert_num=model_args.expert_num,
                target_layers=model_args.lora_target_layers.split(","),
                #
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=args.lora_target_modules.split(",")
                if args.lora_target_modules != "all-linear"
                else args.lora_target_modules,
            )

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing

    return model, peft_config


def create_datasets(tokenizer, model_args, data_args, training_args, train=False):
    if model_args.personal_type in ['profile']:  # ['profile', 'uid']:
        if model_args.peft_type == 'MOE':
            my_dataset = load_dataset4plora_moe(model_args, data_args, training_args, tokenizer)
        else:
            my_dataset = load_dataset4plora_profile(model_args, data_args, training_args, tokenizer, tokenize=False)
    elif model_args.personal_type in ['history']:
        my_dataset = load_dataset4pplug_history(model_args, data_args, training_args, tokenizer)
    elif model_args.personal_type in ['uid']:
        my_dataset = load_dataset4plora_id(model_args, data_args, training_args, tokenizer)
    else:
        print(f'No implementation for {model_args.personal_type}')

    train = True

    spt = '**Answer**'

    def apply_chat_format(sample, in_text='source', out_text='target'):
        dataset_text_field = training_args.dataset_text_field  # text
        if not train:  # infer mode
            chat_text = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample[in_text]},
            ]
        else:
            chat_text = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample[in_text]},
                {"role": "assistant", "content": spt + sample[out_text]}
            ]
        return {dataset_text_field: chat_text}

    def apply_chat_template(samples):
        target = training_args.dataset_text_field
        batch = []
        for tt in samples[target]:
            batch.append(tokenizer.apply_chat_template(
                tt,  # system prompt answer
                max_length=training_args.max_seq_length,
                add_generation_prompt=False if train else False,  # 只有在推理时才添加
                tokenize=False
            ))
        return {target: batch}

    # p, source target
    my_dataset = my_dataset.map(apply_chat_format)  # input, output => chat dict
    my_dataset = my_dataset.map(apply_chat_template, batched=True)  # dict 2 chat template

    # load tokenizer.chat_template
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    # TODO 之训练 complete会过拟合
    collator = None

    return my_dataset, collator


def create_tokenizer(model_args):
    args = model_args
    special_tokens = None
    chat_template = None
    # if args.chat_template_format == "chatml":
    #     special_tokens = ChatmlSpecialTokens
    #     chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    # elif args.chat_template_format == "zephyr":
    #     special_tokens = ZephyrSpecialTokens
    #     chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            # trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template

        # TODO  完全分片数据并行 (FSDP)
        # uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
        # uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP").lower() == "true"
        # if (bnb_config is not None) and uses_fsdp and uses_transformers_4_46:
        #     model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        # else:
        #     model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, )
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def main():
    from args import process_args, make_exp_dirs
    model_args, data_args, training_args = process_args(make_dir=False)

    if model_args.peft_type == 'LORA':
        from trl import SFTTrainer
    elif model_args.peft_type == 'PLORA':
        from trainer.sft_trainer import SFTTrainer, SFTConfig

    tokenizer = create_tokenizer(model_args)

    model, peft_config = create_and_prepare_model(model_args, data_args, training_args)

    my_dataset, collator = create_datasets(tokenizer, model_args, data_args, training_args, train=True)

    make_exp_dirs(model_args, data_args, training_args)  # 在加载 trainer之前

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=my_dataset['train'],
        eval_dataset=my_dataset['dev'],
        peft_config=peft_config,
        data_collator=collator,
    )

    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


def plora_eval(cpt_dict):
    """
    确认  对于instruct model
    input 需要 apply chat template
    padding 一般left_padding (好像没啥影响) , 输入 需要 input_ids 和 attention_mask
    输出需要截断 input_tokens

    with reason&uid
    --adapter_with_name_path

    with reason & profile
    """
    from args import process_args
    model_args, data_args, training_args = process_args()

    model_args.adapter_name_or_path = cpt_dict[data_args.dataset_name]

    metric_func = load_metrics(data_args.dataset_name)

    # tokenizer
    tokenizer = create_tokenizer(model_args)
    print(tokenizer.padding_side)
    tokenizer.padding_side = 'left'  # 'left'

    # datasets [p, input, output, text]
    # TODO  添加 {task prompt , answer prompt , pivot sample, sample num, user profile, score distribution}
    # train_dataset, eval_dataset = create_datasets(tokenizer, model_args, data_args, training_args, train=False)
    my_dataset, collator = create_datasets(tokenizer, model_args, data_args, training_args, train=True)
    # type(model) <class 'plms.llama.LlamaForCausalLM'>
    model, peft_config = create_and_prepare_model(model_args, data_args, training_args)
    # model = get_peft_model(model, peft_config)  # 创建
    model = PeftModel.from_pretrained(model, model_args.adapter_name_or_path)
    model.eval()

    # x['text'] (with apply_chat_template) but with answer
    # padding max_length, longest
    # inputs = tokenizer([x['text'] for x in eval_dataset], return_tensors="pt", padding=True).to(model.device)
    # input_ids, masks = inputs.input_ids, inputs.attention_mask
    ps = torch.tensor([x['p'] for x in my_dataset['dev']], device=model.device)

    outputs = []
    for i, p in tqdm(enumerate(ps)):
        inputs = tokenizer(my_dataset['dev'][i]['text'], return_tensors="pt").to(model.device)
        input_ids, masks = inputs.input_ids, inputs.attention_mask

        output = model.generate(
            input_ids=input_ids,  # .unsqueeze(0),
            attention_mask=masks,  # .unsqueeze(0),
            p=p.unsqueeze(0),  # TODO
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs.append(torch.squeeze(output)[len(input_ids):])

    # TODO 截断  计算非paddings长度
    # outputs = [o[len(i):] for o, i in zip(outputs, input_ids)]
    labels = [x['output'] for x in my_dataset['dev']]
    ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = [{'target': label, 'pred': pred} for label, pred in zip(labels, ans)]
    result_metric = metric_func(results)
    print(result_metric)


def lora_eval(cpts):  # Lora or direct
    """
    --adapter_name_or_path
    # ../output/LaMP_7/SFT_LORA/contriever_Meta-Llama-3-8B-Instruct_lr_0.0001_20241105-161827/checkpoint-200

    ../output/LaMP_3/SFT_PLORA/contriever_Meta-Llama-3-8B-Instruct_lr_5e-05_uid20241106-154134


    --adapter_name_or_path
    /data/mxl/llm/LLaMa_Factory/src/lamp/saves/LaMP_3_with_reason&profile/Meta-Llama-3-8B-Instruct_sft_lora/checkpoint-150

    ../output/LaMP_3/SFT_PLORA/contriever_Meta-Llama-3-8B-Instruct_lr_5e-05_profile20241106-190209

    :return:
    """
    from args import process_args
    model_args, data_args, training_args = process_args()

    tokenizer = create_tokenizer(model_args)
    tokenizer.padding_side = 'left'  # 'left'

    model_args.adapter_name_or_path = cpts[data_args.dataset_name]
    print(f'load from {model_args.adapter_name_or_path}')

    # TODO  添加 {task prompt , answer prompt , pivot sample, sample num, user profile, score distribution}
    my_dataset, collator = create_datasets(tokenizer, model_args, data_args, training_args, train=False)
    inputs = [x['source'] for x in my_dataset['dev']]

    # from LaMP.LaMP.llm_utils import VllmAGent

    llm = VllmAGent(
        model_name_or_path=model_args.model_name_or_path,
        adapter_name_or_path=[model_args.adapter_name_or_path] if model_args.adapter_name_or_path is not None else None
    )
    pd = {
        'use_beam_search': True,
        'best_of': 3,  # >1
        'temperature': 0,
        'top_p': 1,
        'top_k': -1,
        'max_tokens': 512,
        'length_penalty': 0,  # <0 鼓励长句子， >0 鼓励短句子
        'n': 1,  # num_return_sequences=repeat_n,}
    }
    pd = {
        # n=1,
        # max_tokens=512,  # max_new_tokens=128,
        # temperature=0.3,  # 0 趋向于高频词，容易重复， 1< 原始 softmax, >1 区域均匀，随机
        # top_k=20,  # 基于topK单词概率约束候选范围
        # length_penalty=1,  # <0 鼓励长句子， >0 鼓励短句子

        'temperature': 0, 'top_p': 1, 'top_k': -1, 'max_tokens': 512, 'length_penalty': 1, 'n': 1,
    }
    responses = llm.infer_vllm(inputs, instruction=None, chat=True, param=pd)  # 训练数据已经添加了chat_template

    labels = [x['target'] for x in my_dataset['dev']]  # p source target text
    results = [{'target': label, 'pred': pred} for label, pred in zip(labels, responses)]

    metric_func = load_metrics(data_args.dataset_name)
    result_metric = metric_func(results)
    print(result_metric)


if __name__ == "__main__":
    # TODO  在 T5 和 较小的模型验证完毕后使用 llama 进行 实验

    """
    简单修改 prompt complete loss only
    ../output/LaMP_1/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241214-234201
    ../output/LaMP_3/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241214-235922
    ../output/LaMP_4/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241215-012819
    ../output/LaMP_5/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241215-014324
    ../output/LaMP_7/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241215-021709
    """
    # main()
    cpt_dict = {
        'LaMP_3': '../output/LaMP_3/SFT_PLORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241218-171606'

    }

    plora_eval(cpt_dict)
    # cpt_dict = {
    #     'LaMP_1': '../output/LaMP_1/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241216-105241',
    #     # 'LaMP_3': '../output/LaMP_3/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241216-111542',
    #     'LaMP_3': '../output/LaMP_3/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241218-100513',
    #     'LaMP_4': '../output/LaMP_4/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241216-123901',
    #     'LaMP_5': '../output/LaMP_5/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241216-131207',
    #     'LaMP_7': '../output/LaMP_7/SFT_LORA/contriever_Llama-3.2-1B-Instruct_lr_0.0005_profile_20241216-135449',
    # }
    # 存在严重复读机问题  => train complete only
    # lora_eval(cpt_dict)

    # TODO 考虑推理加速 （合并plora参数，自定义vllm）
    # 自定义vllm https://docs.vllm.ai/en/latest/models/adding_model.html

    model2path = {
        'bert': '',
        'roberta': '',
        'flan_t5': '/data/hf/google/flan-t5-base',
        # 'llama': '/data/hf/meta-llama/Meta-Llama-3-8B-Instruct',
        'llama': '/data/hf/meta-llama/Llama-3.2-1B-Instruct',
    }

    """
    --use_reentrant
    True
    --use_4bit_quantization
    True
    --use_nested_quant
    True
    --bnb_4bit_compute_dtype
    bfloat16
    --fp16
    True
    
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj","down_proj"]
    
    ###########################
    
    --personal_type  uid, profile
    
    --adapter_name_or_path
    ../output/LaMP_3/SFT_PLORA/contriever_Llama-3.2-1B-Instruct_lr_5e-05_20241101-005322/checkpoint-150
    --adapter_name_or_path
    ../output/LaMP_3/SFT_PLORA/contriever_Meta-Llama-3-8B-Instruct_lr_5e-05_20241102-161628
    
    # plora with more prompt
    --personal_type uid
    --adapter_name_or_path
    ../output/LaMP_3/SFT_PLORA/contriever_Meta-Llama-3-8B-Instruct_lr_0.0001_20241103-133353/checkpoint-450
    
    # plora - profile
    --personal_type profile
    --adapter_name_or_path
    ../output/LaMP_3/SFT_PLORA/contriever_Meta-Llama-3-8B-Instruct_lr_0.0001_20241103-162943/checkpoint-150
    # {'mae': 0.4728, 'rmse': 0.9562426470305536}
    
    # plora -profile
    # with profile 1sample socre_dist input detail_prompt => direct output
    --personal_type profile
    --adapter_name_or_path 
    ../output/LaMP_3/SFT_LORA/contriever_Meta-Llama-3-8B-Instruct_lr_0.0001_20241104-144335/checkpoint-300
    
    --seed 100
    --model_name_or_path '/data/hf/meta-llama/Llama-3.2-1B-Instruct',
    --dataset_name "LaMP_3"
    --dataset_text_field "input"
    --output_dir './output'
    --train_method 'sft'
    --peft_type lora
    --lora_r 8
    --lora_alpha 16
    --lora_dropout 0.1
    --lora_target_modules "all-linear"
    --add_special_tokens False
    --append_concat_token False
    --max_seq_len 512
    --num_train_epochs 1
    --logging_steps 5
    --log_level "info"
    --logging_strategy "steps"
    --eval_strategy "epoch"
    --save_strategy "epoch"
    --bf16 True
    --packing True
    --learning_rate 1e-4
    --lr_scheduler_type "cosine"
    --weight_decay 1e-4
    --warmup_ratio 0.0
    --max_grad_norm 1.0
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 8
    --gradient_accumulation_steps 8
    
    --gradient_checkpointing True
    --use_reentrant True
    
    --use_4bit_quantization True
    --use_nested_quant True
    --bnb_4bit_compute_dtype "bfloat16"
    
    --use_flash_attn True
    """
