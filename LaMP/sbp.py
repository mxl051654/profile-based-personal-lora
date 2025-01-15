import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import datasets
from tqdm import tqdm
from args import load_args
from collections import defaultdict

sum_items = {
    "default": """
                Keywords: [keyword1, keyword2, . . . ]
                Topics: [topic1, topic2, . . . ]
                Writing Style: [style1, style2, . . . ]
                Preferences: [preference1, preference2, . . . ]
                """,
    "LaMP_t_1": "Keywords: [keyword1, keyword2, keyword3,. . . ]\n"
                "Topics: [topics1, topics2, topics3, . . . ]",
    "LaMP_t_2": "Keywords: [keyword1, keyword2, keyword3,. . . ]\n"
                "Topics: [topics1, topics2, topics3, . . . ]",
    "LaMP_t_3": "Rating Distribution: {score dict}\n"
                "Rating Patterns: [pattern1, pattern2, . . . ]",
    "LaMP_t_4": "Writing Style: [style1, style2, ...]\n"
                "Content Patterns: [patterns1, patterns2, ...]",
    "LaMP_t_5": "Writing Style: [style1, style2, ...]\n "
                "Title Patterns: [pattern1, pattern2, ...]",
    "LaMP_t_6": "Keywords: [keyword1, keyword2, keyword3, ...]\n"
                "Topics: [topics1, topics2, topics3, ...]",
    "LaMP_t_7": "Writing Style: [style1, style2, ...]\n  "
                "Tone: [tone1, tone2, ...]\n "
                "length: [length1, length2, ...]",

    "LaMP_1": "Keywords: [keyword1, keyword2, keyword3,. . . ]\n"
              "Topics: [topics1, topics2, topics3, . . . ]",
    "LaMP_2": "Keywords: [keyword1, keyword2, keyword3,. . . ]\n"
              "Topics: [topics1, topics2, topics3, . . . ]",
    "LaMP_3": "Rating Distribution: {score dict}\n"
              "Rating Patterns: [pattern1, pattern2, . . . ]",
    "LaMP_4": "Writing Style: [style1, style2, ...]\n"
              "Content Patterns: [patterns1, patterns2, ...]",
    "LaMP_5": "Writing Style: [style1, style2, ...]\n "
              "Title Patterns: [pattern1, pattern2, ...]",
    "LaMP_6": "Keywords: [keyword1, keyword2, keyword3, ...]\n"
              "Topics: [topics1, topics2, topics3, ...]",
    "LaMP_7": "Writing Style: [style1, style2, ...]\n  "
              "Tone: [tone1, tone2, ...]\n "
              "length: [length1, length2, ...]",

    "IMDB-B": "Rating Distribution: {score dict}\n"
              "Rating Patterns: [pattern1, pattern2, . . . ]",
    "YELP-B": "Rating Distribution: {score dict}\n"
              "Rating Patterns: [pattern1, pattern2, . . . ]",
    "GDRD-B": "Rating Distribution: {score dict}\n"
              "Rating Patterns: [pattern1, pattern2, . . . ]",
    "PPR-B": "Rating Distribution: {score dict}\n"
             "Rating Patterns: [pattern1, pattern2, . . . ]",
}

profile_sum_templates = {
    'sbp': """
            **User History**
            I will provide you with some examples of the
            user’s past interactions. Please analyze these to
            create a user profile. Each example consists of an
            input and the corresponding output.
            {{examples}}
            **Instruction**
            Based on the provided examples, please generate
            a user profile in the following format:
            {{task_format}}""",  # + sum_items[task],
    'hierarchy_first': """
            **User History**
            I will provide you with some examples of the
            user’s past interactions. Please analyze these to
            create a user profile. Each example consists of an
            input and the corresponding output.
            {{examples}}
            **Instruction**
            Based on the provided examples, please generate
            a user profile in the following format:
            {{task_format}}""",  # + sum_items[task],,
    'hierarchy_next': """
            **User profile History**
            I will provide you with some examples of the
            user’s past profiles. Please analyze these to
            create a new accurate and concise profile. 
            {{examples}}
            **Instruction**
            Based on the provided profiles, please generate
            a new accurate and concise profile. in the following format:
            {{task_format}}""",  # + sum_items[task],,
    'recurrent_first': '',
    'recurrent_next': '',
}


def apply_template(profile, review_text, score_most=5):
    """
    Now you have written this new product review:
    Product Review: {{review_text}}
    """
    lamp_3_template = Template("""
        User Profile
        Assuming you have written headlines with the
        following characteristics:{{profile}}
        Rating Task: {{review_text}}
        Based on the review, please analyze its sentiment
        and how much you like the product.
        Instruction
        Follow your previous rating habits and these instructions:
        • If you feel satisfied with this product or have
        concerns but it’s good overall, it should be rated 5.
        • If you feel good about this product but notice
        some issues, it should be rated as 4.
        • If you feel OK but have concerns, it should be rated as 3.
        • If you feel unsatisfied with this product but it’s
        acceptable for some reason, it should be rated as 2.
        • If you feel completely disappointed or upset, it
        should be rated 1.
        Your most common rating is {{score_most}}.
        You must follow this rating pattern faithfully 
        and answer with the rating without further explanation.
        """.replace('\t', ''))
    return lamp_3_template.render(profile=profile, review_text=review_text, scoremost=score_most)


import random
from jinja2 import Template
import re
import json
import jsonlines
import evaluate
import nltk

nltk.data.path.append('/data/mxl/nltk_data')
# pip install --upgrade datasets multiprocess
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator

from llm_utils import VllmAGent
from metrics.my_metrics import load_metrics, cal_metric4lamp_3


def load_dataset(args):
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    # collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=args.max_length)

    task = args.task
    if args.use_profile:  # prompt complete with profile
        prompt_generator, contriver = create_prompt_generator(args.num_retrieved, args.retriever, args.is_ranked,
                                                              args.max_length, tokenizer)
    else:
        prompt_generator, contriver = None, None

    greater_is_better = True
    metric_func = load_metrics(args.task)
    print('process dataset')
    if task in ["LaMP_1", "LaMP_t_1"]:
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer, all_labels=labels)
        best_metric = "accuracy"
    elif task == "LaMP_2-old":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer, all_labels=labels)
        best_metric = "accuracy"
    elif task == "LaMP_2":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer, all_labels=labels)
        best_metric = "accuracy"
    elif task in ["LaMP_3", 'LaMP_t_3', 'IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func
        best_metric = "mae"
        greater_is_better = False
    elif task in ["LaMP_4", 'LaMP_t_4']:
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task in ["LaMP_5", "LaMP_t_5"]:
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task in ["LaMP_7", "LaMP_t_7"]:
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP_6":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer)
        best_metric = "rouge-1"

    # dataset id source target
    # dataset.data  id input profile{id text score} output
    dataset = {'train': train_dataset, 'dev': eval_dataset, 'test': test_dataset}
    return dataset, compute_metrics, best_metric


def sum_profile():
    def apply_sum_template(profiles, tokenizer, max_length, task):
        ttext = """
            **User History**
            I will provide you with some examples of the
            user’s past interactions. Please analyze these to
            create a user profile. Each example consists of an
            input and the corresponding output.
            {{examples}}
            Instruction
            Based on the provided examples, please generate
            a user profile in the following format:
            """ + sum_items[task]

        sum_template = Template(ttext)
        examples = '\n'.join(
            [f"Example {i}\n" + '\n'.join([f"{k} : {v}" for k, v in x.items()]) for i, x in enumerate(profiles)])

        # 截断
        tokens = tokenizer(examples, max_length=max_length - len(tokenizer(ttext)), truncation=True)
        examples = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)

        return sum_template.render(examples=examples)

    args = load_args(make_dir=True)

    dataset, _, _ = load_dataset(args)
    dataset = dataset[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    inputs = [apply_sum_template(
        x['profile'],
        tokenizer,
        max_length=args.max_length,
        task=args.task) for x in dataset.data]

    # infer profiles
    llm = VllmAGent(model_name_or_path=args.model_name)
    results = llm.infer_vllm(inputs, chat=True)

    with jsonlines.open(f'{args.profile_dir}/{args.profile_path}', mode='w') as f:
        f.write_all([{
            'input': x,
            'output': y,
        } for x, y in zip(inputs, results)])


def sum_with_div_profile():
    def sample_profiles_by_score(profiles, n):
        # TODO 后续其他数据考虑使用 k-mean 扩展多样性
        # 1. 按 score 分组
        grouped_profiles = defaultdict(list)
        for profile in profiles:
            grouped_profiles[profile['score']].append(profile)

        # 2. 计算每个 score 的最小采样量
        unique_scores = list(grouped_profiles.keys())
        min_samples_per_score = int(n / len(unique_scores))

        sampled_profiles = []

        # 3. 对每个 score 进行采样
        for score, group in grouped_profiles.items():
            if len(group) <= min_samples_per_score:
                # 如果样本不足，采样所有
                sampled_profiles.extend(group)
            else:
                # 随机采样
                sampled_profiles.extend(random.sample(group, min_samples_per_score))

        # 4. 如果总采样数不足 n，继续从剩余数据中补充
        remaining_samples_needed = n - len(sampled_profiles)
        if remaining_samples_needed > 0:
            remaining_profiles = [p for score, group in grouped_profiles.items() for p in group if
                                  p not in sampled_profiles]
            if len(remaining_profiles) > remaining_samples_needed:
                sampled_profiles.extend(random.sample(remaining_profiles, remaining_samples_needed))
            else:
                sampled_profiles.extend(remaining_profiles)
        return sampled_profiles

    def cal_dist(histories):
        #  profile[id, score(str),text]
        score_counts = defaultdict(int)  # 初始化一个默认整数计数器
        for profile in histories:
            score = profile['score']
            score_counts[score] += 1  # 每遇到一个分数就加一
        return str(dict(score_counts))  # 转换成普通的字典返回

    def apply_sum_template_with_dist(profiles, tokenizer, max_length, task, dist):
        ttext = """
            **User History**
            I will provide you with some examples of the
            user’s past interactions. Please analyze these to
            create a user profile. Each example consists of an
            input and the corresponding output.
            {{examples}}
            **User Score Distribution**
            {{dist}}
            Instruction
            Based on the provided examples, please generate
            a user profile in the following format:
            """ + sum_items[task]

        def clean_text(text):
            return text.replace('\n', '')

        sum_template = Template(ttext)

        # 拼接全部 并截断
        examples = '\n'.join([f"Example {i}\n"
                              f"Input:{clean_text(x['text'])}\n"
                              f"Output:{x['score']}" for i, x in enumerate(profiles)])

        # 截断
        tokens = tokenizer(examples, max_length=max_length - len(tokenizer(ttext)), truncation=True)
        examples = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)

        return sum_template.render(examples=examples, dist=dist)

    args = load_args(make_dir=True)

    assert args.task == 'LaMP_3'

    dataset, _, _ = load_dataset(args)
    dataset = dataset[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dists = [cal_dist(x['profile']) for x in dataset.data]

    inputs = [apply_sum_template_with_dist(
        sample_profiles_by_score(x['profile'], n=5),
        tokenizer,
        max_length=args.max_length,
        task=args.task,
        dist=dists[i]
    ) for i, x in enumerate(dataset.data)]

    # infer profiles
    llm = VllmAGent(model_name_or_path=args.model_name)
    results = llm.infer_vllm(inputs, chat=True)

    with jsonlines.open(f'{args.profile_dir}/{args.profile_path}', mode='w') as f:
        f.write_all([{
            'input': x,
            'output': y,
        } for x, y in zip(inputs, results)])


def reflect_opt_profile():
    """ pip line
     (1)  随机从历史数据采样 m 个样本
    （2）推理并计算指标
    （3）指标低于阈值则进行 反思
     """
    args = load_args(make_dir=True)

    dataset, _, _ = load_dataset(args)
    dataset = dataset[args.split]
    llm = VllmAGent(model_name_or_path=args.model_name)

    profiles_raw = list(jsonlines.open(f'{args.profile_dir}/{args.profile_path}'.replace('Iter', 'SBP'), 'r'))
    profiles = [x['output'] for x in profiles_raw]

    for i, item in enumerate(dataset.data):
        profile = profiles[i]
        # sample
        sample_history = random.sample(item['profile'], 5)  # {id, text, score}

        # infer
        lamp_3_template = '"What is the score of the following review on a scale of 1 to 5? \
            just answer with 1, 2, 3, 4, or 5 without further explanation. review: '
        inputs = [apply_template(profile, lamp_3_template + x['text']) for x in sample_history]
        result = llm.infer_vllm(inputs, chat=True)
        result = [{'input': x['text'],
                   'target': x['score'],
                   'pred': p} for x, p in zip(sample_history, result)]

        # eval
        metrics, error_sample = cal_metric4lamp_3(result, return_error=True)

        # get error sample
        print(f'Error num {len(error_sample)}')
        for e in error_sample:
            error = f'Input:{e["input"]} True Answer:{e["target"]} Error Pred:{e["pred"]}'

            reflect_template = Template("""
            Here is a user preference profile and a error infer result base on this profile,
            you need to reflect on this error and optimize the profile.
            **Profile**
            {{profile}}
             **User History**
            {{error}}
            **Instruction**
            Based on the provided examples, generate
            a better user profile in the following format:
            """ + sum_items[args.task])
            reflect_input = [reflect_template.render(profile=profile, error=error)]
            profile = llm.infer_vllm(reflect_input, chat=True)
            profiles[i] = profile

    # reflect

    with jsonlines.open(f'{args.profile_dir}/{args.profile_path}', mode='w') as f:
        f.write_all([{
            'input': x['input'],
            'output': y,
        } for x, y in zip(profiles_raw, profiles)])


def hierarchy_sum_profile():
    def clean_text(text):
        return text.replace('\n', '')

    def apply_sum_template(template, profiles, tokenizer, max_length, task):

        sum_template = Template(template)
        # examples = '\n'.join([f"Example {i}\n"
        #                       f"Input:{clean_text(x['text'])}\n"
        #                       f"Output:{x['score']}" for i, x in enumerate(profiles)])

        # 截断(默认是和input相似的topk)
        tokens = tokenizer(profiles, max_length=max_length - len(tokenizer(template)), truncation=True)
        examples = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)

        return sum_template.render(examples=examples, task_format=sum_items[task])

    args = load_args(make_dir=True)

    dataset, _, _ = load_dataset(args)
    dataset = dataset[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = VllmAGent(model_name_or_path=args.model_name)

    # split  split->user map
    def split_profile(tokenizer, profile_list, max_tokens=1024):
        split_profiles = []
        current_chunk = []

        current_length = 0
        for profile in profile_list:
            tokens = tokenizer.tokenize(profile)
            token_length = len(tokens)

            # 检查当前profile是否能放入当前的chunk
            if current_length + token_length > max_tokens:
                split_profiles.append(current_chunk)  # 保存当前chunk
                current_chunk = []  # 开启新的chunk
                current_length = 0

            current_chunk.append(profile)
            current_length += token_length

        if current_chunk:
            split_profiles.append(current_chunk)  # 保存最后一个chunk

        return split_profiles

    # sum
    profiles = [x['profile'] for x in dataset.data]
    sums = []
    for profile in profiles:
        profile = [f"Input:{clean_text(x['text'])}\n"
                   f"Output:{x['score']}" for i, x in enumerate(profile)]
        for i in range(20):
            if len(profile) < 2:
                break
            task_format = sum_items[args.task]
            h_f_template = profile_sum_templates['hierarchy_first'] if i == 0 else \
                profile_sum_templates['hierarchy_next']
            left_len = len(tokenizer.tokenize(''.join([task_format, h_f_template])))
            # 合并
            split_user_profiles = split_profile(tokenizer, profile,
                                                max_tokens=args.max_length - left_len)

            inputs = [
                apply_sum_template(h_f_template, ''.join([f'Sample {i}\n {p_item}\n' for i, p_item in enumerate(p)]),
                                   tokenizer,
                                   max_length=args.max_length,
                                   task=args.task) for p in split_user_profiles]

            profile = llm.infer_vllm(inputs, chat=True)

        sums.append(profile[0])

    with jsonlines.open(f'{args.profile_dir}/{args.profile_path}', mode='w') as f:
        f.write_all([{
            'input': x,
            'output': y,
        } for x, y in zip(profiles, sums)])


def recurrent_sum_profile():
    def clean_text(text):
        return text.replace('\n', '')

    def apply_sum_template(template, profiles, tokenizer, max_length, task):

        sum_template = Template(template)
        # examples = '\n'.join([f"Example {i}\n"
        #                       f"Input:{clean_text(x['text'])}\n"
        #                       f"Output:{x['score']}" for i, x in enumerate(profiles)])

        # 截断(默认是和input相似的topk)
        tokens = tokenizer(profiles, max_length=max_length - len(tokenizer(template)), truncation=True)
        examples = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)

        return sum_template.render(examples=examples, task_format=sum_items[task])

    args = load_args(make_dir=True)

    dataset, _, _ = load_dataset(args)
    dataset = dataset[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = VllmAGent(model_name_or_path=args.model_name)

    # split  split->user map
    def split_profile(tokenizer, profile_list, max_tokens=1024):
        split_profiles = []
        current_chunk = []

        current_length = 0
        for profile in profile_list:
            tokens = tokenizer.tokenize(profile)
            token_length = len(tokens)

            # 检查当前profile是否能放入当前的chunk
            if current_length + token_length > max_tokens:
                split_profiles.append(current_chunk)  # 保存当前chunk
                current_chunk = []  # 开启新的chunk
                current_length = 0

            current_chunk.append(profile)
            current_length += token_length

        if current_chunk:
            split_profiles.append(current_chunk)  # 保存最后一个chunk

        return split_profiles

    # sum
    profiles = [x['profile'] for x in dataset.data]
    sums = []
    for profile in tqdm(profiles):
        profile = [f"Example {i}\n"
                   f"Input:{clean_text(x['text'])}\n"
                   f"Output:{x['score']}" for i, x in enumerate(profile)]
        for i in range(10):
            if len(profile) < 2:
                break
            task_format = sum_items[args.task]
            h_f_template = profile_sum_templates['hierarchy_first'] if i == 0 else \
                profile_sum_templates['hierarchy_next']
            left_len = len(tokenizer.tokenize(''.join([task_format, h_f_template])))
            # 合并
            split_user_profiles = split_profile(tokenizer, profile,
                                                max_tokens=args.max_length - left_len)

            inputs = [apply_sum_template(h_f_template, p, tokenizer,
                                         max_length=args.max_length,
                                         task=args.task) for p in split_user_profiles]

            profile = llm.infer_vllm(inputs, chat=True)

        sums.append(profile[0])

    with jsonlines.open(f'{args.profile_dir}/{args.profile_path}', mode='w') as f:
        f.write_all([{
            'input': x,
            'output': y,
        } for x, y in zip(profiles, sums)])


def infer_main():
    args = load_args(make_dir=True)
    dataset, _, _ = load_dataset(args)
    dataset = dataset[args.split]

    inputs = []
    if args.method in ['SBP', 'SBP_div', 'Iter', 'Hierarchy']:
        profiles = list(jsonlines.open(f'{args.profile_dir}/{args.profile_path}', 'r'))
        profiles = [x['output'] for x in profiles]
        inputs = [apply_template(args.task, p, x['input']) for p, x in zip(profiles, dataset.data)]

    # model_path = '/data/hf/meta-llama/Meta-Llama-3-8B-Instruct'
    llm = VllmAGent(model_name_or_path=args.model_name)
    result = llm.infer_vllm(inputs, chat=True)

    with jsonlines.open(f'{args.output_dir}/{args.profile_path}', mode='w') as f:
        f.write_all([{
            'id': x['id'],
            'source': inputs[i],
            'target': x['target'],
            'pred': result[i],
        } for i, x in enumerate(dataset)])

    ################## Eval ########################
    with jsonlines.open(f'{args.output_dir}/{args.profile_path}', mode='r') as f:
        result = list(f)

    print(cal_metric4lamp_3(result))


def eval(result_dir):
    def extract_numbers(text):
        # 使用正则表达式查找文本中的数字
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

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


if __name__ == '__main__':
    # TODO: Direct => SBP
    sum_profile()

    # TODO with diversity sample
    # sum_with_div_profile()

    # TODO: 基于 history 数据 进行校验和反思优化 profile  Iter=>Reflect
    # iter_opt_profile()

    # TODO: 对比 迭代/层次摘要生成 方式
    # hierarchy_sum_profile()
    """
    nohup python sbp.py --task LaMP_3 \
    --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct \
    --retriever contriever \
    --use_profile \
    --is_ranked \
    --max_length 1024 \
    --method SBP_div &
    """
