"""
direct prompt
ICL prompt
cot prompt
StepBack prompt
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from jinja2 import Template
import re
import jsonlines
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

import argparse
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor

from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator
from datetime import datetime


from llm_utils import VllmAGent


def load_args(make_dir=False):
    """
    --model_name /data/hf/google/flan-t5-base
    --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct

    --task LaMP_3
    --train_data ./rank/LaMP_3/train_questions_rank_bm25_merge.json
    --validation_data ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --test_data ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
    --retriever bm25
    --use_profile
    --is_ranked
    --num_retrieved 4
    --method Direct
    """

    parser = argparse.ArgumentParser()

    T = 4
    parser.add_argument("--train_data", default=f'./rank/LaMP_{T}/train_questions_rank_bm25_merge.json')
    parser.add_argument("--validation_data", default=f'./rank/LaMP_{T}/dev_questions_rank_bm25__merge.json')
    parser.add_argument("--test_data", default=f'./rank/LaMP_{T}/dev_questions_rank_bm25_merge.json')
    # 不开放测试集，上传结果进行测试

    parser.add_argument("--model_path", default='/data/hf/google/flan-t5-base')
    parser.add_argument("--adapter_path", default='/data/hf/google/flan-t5-base')
    parser.add_argument("--task", default=f'LaMP_{T}', )  # choices=[f'LaMP_{i}' for i in range(1, 8)])
    parser.add_argument("--method", default='Direct',
                        choices=['SFT',
                                 'SFT_with_history', 'SFT_with_prompt', 'SFT_with_profile',
                                 'Direct', 'SBP'])

    # ../ouput/[task]/[LaMP,ROPG,RSPG]/exp_paras / [logs, cpts, results]
    parser.add_argument("--output_dir", default='./output')  # 存放单次实验 cpt和结果

    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--use_profile", action="store_true")
    parser.add_argument("--is_ranked", action="store_true")
    parser.add_argument("--num_retrieved", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=1024)  # 太短会导致profile被截断，参考数据统计
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--generation_num_beams", type=int, default=4)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--cache_dir", default="./cache")

    args = parser.parse_args()

    # 对不同任务/方法，设计不同的参数
    # direct  k max_len

    args.train_data = f'./rank/{args.task}/train_questions_rank_{args.retriever}_merge.json'
    args.validation_data = f'./rank/{args.task}/dev_questions_rank_{args.retriever}_merge.json'
    args.test_data = f'./rank/{args.task}/dev_questions_rank_{args.retriever}_merge.json'

    paras = f"{args.retriever}_{args.model_path.split('/')[-1]}_" \
            f"k_{args.num_retrieved}_maxlen_{args.max_length}_" + \
            datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = f"../output/{args.task}/{args.method}/{paras}"
    print(f'Experiment Path :\t {args.output_dir}')

    if make_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        for dn in ['cpts', 'logs', 'results', 'best_cpt']:
            if not os.path.exists(f"{args.output_dir}/{dn}"):
                os.mkdir(f"{args.output_dir}/{dn}")
                print(f'make dir {args.output_dir}/{dn}')

    return args


def load_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    # collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=args.max_length)

    task = args.task
    if args.use_profile:  # prompt complete with profile
        prompt_generator, contriver = create_prompt_generator(args.num_retrieved, args.retriever, args.is_ranked,
                                                              args.max_length, tokenizer)
    else:
        prompt_generator, contriver = None, None

    greater_is_better = True
    print('process dataset')
    if task == "LaMP_1":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
        best_metric = "accuracy"
    elif task == "LaMP_2-old":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
        best_metric = "accuracy"
    elif task == "LaMP_2":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
        best_metric = "accuracy"
    elif task == "LaMP_3":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels)
        best_metric = "mae"
        greater_is_better = False
    elif task == "LaMP_4":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP_5":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP_7":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP_6":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
        best_metric = "rouge-1"

    # dataset id source target
    # dataset.data  id input profile{id text score} output
    # TODO: 修改metric
    print()
    dataset = {'train': train_dataset, 'dev': eval_dataset, 'test': test_dataset}
    return dataset, compute_metrics, best_metric


def sft_infer():
    """
    --task LaMP_3
    --method SFT_with_history
    --train_data ./rank/LaMP_3/train_questions_rank_bm25_merge.json
    --validation_data ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --test_data ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --model_path /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
    --adapter_path /data/mxl/llm/LLaMa_Factory/src/lamp/saves/LaMP_3_with_history/Meta-Llama-3-8B-Instruct_sft_lora
    --retriever contriever
    --use_profile
    --is_ranked
    --num_retrieved 4
    --max_length 1024
    """
    args = load_args(make_dir=True)
    # dataset, _, _ = load_dataset(args)
    save_dir = f'{args.output_dir}/results'

    split = 'dev'
    # if args.method == 'SFT_with_history':
    # with prompt (no hisotry)
    # save_data = [{'instruction': '', 'input': x['input'], 'output': x['output']}
    #              for x in dataset['train'].data] + \
    #             [{'instruction': '', 'input': x['input'], 'output': x['output']}
    #              for x in dataset['dev'].data]

    # wit history
    # data = [{'instruction': '', 'input': x['source'], 'output': x['target']}
    #         for x in dataset['dev']]

    train_data_path = '/data/mxl/llm/LLaMa_Factory/data/lamp/LaMP_3_with_profile_train_eval.json'


    dataset = list(jsonlines.open(train_data_path))[0]

    # 统计训练数据分布  sum([a==str(5) for a in [x['output'] for x in dataset[:20000]]])  [690, 824, 1876, 4497, 12113]
    # eval 数据分布  [101, 101, 210, 610, 1478]

    eval_num = 2500
    # inputs = [x['source'] for x in dataset[-eval_num:]]
    # targets = [x['target'] for x in dataset[-eval_num:]]
    inputs = [x['input'] for x in dataset[-eval_num:]]
    targets = [x['output'] for x in dataset[-eval_num:]]

    llm = VllmAGent(model_name_or_path=args.model_path, adapter_name_or_path=[args.adapter_path],)
    train_result = llm.infer_vllm(inputs, chat=True)
    # 对于评分，样本相当不均衡

    with jsonlines.open(f'{save_dir}/{split}_result.jsonl', mode='w') as f:
        f.write_all([{
            'source': inputs[i],
            'target': targets[i],
            'pred': train_result[i],
        } for i in range(len(inputs))])

    ################## Eval #########################

    def extract_numbers(text):
        # 使用正则表达式查找文本中的数字
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

    with jsonlines.open(f'{save_dir}/{split}_result.jsonl', mode='r') as f:
        result = list(f)

    decoded_preds, decoded_labels = [extract_numbers(x['pred']) for x in result], [x['target'] for x in result]

    def create_mapping(x, y):
        try:
            return float(x)  # 可以解析
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):  # 否则取最远的
                return 1.0
            else:
                return 5.0

    metric_dir = '/data/mxl/metrics'
    mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
    mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

    decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
    decoded_labels = [create_mapping(x, x) for x in decoded_labels]
    result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
    result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}

    print(result)


def eval(result_dir):
    def extract_numbers(text):
        # 使用正则表达式查找文本中的数字
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

    # args = load_args()
    # train_dataset, eval_dataset, test_dataset, compute_metrics, best_metric
    # train_dataset, eval_dataset, _, compute_metrics, _ = load_dataset(args)

    # result_dir = '/data/mxl/rlhf/LaMP/output/LaMP_3/Direct/bm25_Meta-Llama-3-8B-Instruct_lr_5e-05_20241004-185726'
    with jsonlines.open(f'{result_dir}/results/train_result.jsonl') as f:
        result = list(f)

    # score = compute_metrics(([extract_numbers(x['pred']) for x in result],
    #                          [x['target'] for x in result]))
    decoded_preds, decoded_labels = [extract_numbers(x['pred']) for x in result], [x['target'] for x in result]

    def create_mapping(x, y):
        try:
            return float(x)  # 可以解析
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):  # 否则取最远的
                return 1.0
            else:
                return 5.0

    metric_dir = '/data/mxl/metrics'
    mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
    mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

    decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
    decoded_labels = [create_mapping(x, x) for x in decoded_labels]
    result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
    result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}

    print(result)


def direct_infer():
    model_path = '/data/hf/meta-llama/Meta-Llama-3-8B-Instruct'
    llm = VllmAGent(model_name_or_path=model_path)

    args = load_args(make_dir=True)
    train_dataset, eval_dataset, _, _, _ = load_dataset(args)

    save_dir = f'{args.output_dir}/results'

    inputs = []

    if args.method == 'Direct':
        inputs = [x['source'] for x in train_dataset]
    # elif args.method == 'SBP':

    train_result = llm.infer_vllm(inputs, chat=True)

    with jsonlines.open(f'{save_dir}/train_result.jsonl', mode='w') as f:
        f.write_all([{
            'id': x['id'],
            'source': x['source'] + '. [Note] direct answer score in format: Score [X] , now answer it, Score ',
            'target': x['target'],
            'pred': y,
        } for x, y in zip(train_dataset, train_result)])

    ################## Eval #########################
    def extract_numbers(text):
        # 使用正则表达式查找文本中的数字
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

    with jsonlines.open(f'{save_dir}/train_result.jsonl', mode='r') as f:
        result = list(f)

    decoded_preds, decoded_labels = [extract_numbers(x['pred']) for x in result], [x['target'] for x in result]

    def create_mapping(x, y):
        try:
            return float(x)  # 可以解析
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):  # 否则取最远的
                return 1.0
            else:
                return 5.0

    metric_dir = '/data/mxl/metrics'
    mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
    mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

    decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
    decoded_labels = [create_mapping(x, x) for x in decoded_labels]
    result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
    result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}

    print(result)


if __name__ == '__main__':
    """
    直接推理系列实验
    
    #公共
    用户公用 挑选的高质量profile
    用户公用  高质量 的 rule
    
    #群组
    聚类  
    
    #个性化
    用户使用检索的profile
    用户使用个性化profile sum
    
    """
    # infer_main()

    # sum_profile()

    sft_infer()

    # result_dir = '/data/mxl/rlhf/LaMP/output/LaMP_3/Direct/bm25_Meta-Llama-3-8B-Instruct_k_2_maxlen_512_20241004-201118'
    # eval(result_dir)

    # bm25
    # k 4 max_len  1024
    # {'mae': 0.49665, 'rmse': 1.004315687421042}
    # k 2 max_len  512
    # {'mae': 0.4807, 'rmse': 0.8434453153583817}
    # k 1 max_len  512
    # {'mae': 0.5291, 'rmse': 0.9143850392476902}

    # contriever
    # k 2 max_len  512
    # {'mae': 0.4378, 'rmse': 0.7794228634059948}

    ##
    """
    /data/mxl/llm/LLaMa_Factory/src/lamp/saves/LaMP_3_with_history/Meta-Llama-3-8B-Instruct_sft_lora
    /data/mxl/llm/LLaMa_Factory/src/lamp/saves/LaMP_3_with_prompt/Meta-Llama-3-8B-Instruct_sft_lora
    
    --task
    LaMP_3
    --method
    SFT_with_history
    --train_data
    ./rank/LaMP_3/train_questions_rank_bm25_merge.json
    --validation_data
    ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --test_data
    ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --model_path
    /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
    --adapter_path
    /data/mxl/llm/LLaMa_Factory/src/lamp/saves/LaMP_3_with_history/Meta-Llama-3-8B-Instruct_sft_lora
    --retriever
    contriever
    --use_profile
    --is_ranked
    --num_retrieved
    4
    --max_length
    1024
    
    
    # for lora with prompt/history/profile
    --task
    LaMP_3
    --method
    SFT_with_history
    --train_data
    ./rank/LaMP_3/train_questions_rank_bm25_merge.json
    --validation_data
    ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --test_data
    ./rank/LaMP_3/dev_questions_rank_bm25_merge.json
    --model_path
    /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
    --adapter_path
    /data/mxl/llm/LLaMa_Factory/src/lamp/saves/LaMP_3_with_profile/Meta-Llama-3-8B-Instruct_sft_lora
    --retriever
    contriever
    --use_profile
    --is_ranked
    --num_retrieved
    4
    --max_length
    1024
    """
