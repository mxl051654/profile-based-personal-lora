"""
copy from LaMP/data/dataset
"""
import os
import json
import random
from collections import defaultdict
from jinja2 import Template
import jsonlines
import torch
from datasets import DatasetDict
from datasets import Dataset as HFDataset
from transformers import AutoModel, AutoTokenizer


def get_all_labels(task):
    # classification
    if task in ["LaMP_1", "LaMP_t_1"]:
        return ["[1]", "[2]"]
    elif task in ["LaMP_2", "LaMP_t_2"]:
        return [
            'sci-fi', 'based on a book', 'comedy', 'action',
            'twist ending', 'dystopia', 'dark comedy', 'classic',
            'psychology', 'fantasy', 'romance', 'thought-provoking',
            'social commentary', 'violence', 'true story'
        ]
    elif task in ["LaMP_3", 'LaMP_t_3']:
        return ["1", "2", "3", "4", "5"]

    # generation
    elif task in ["LaMP_4", 'LaMP_t_4']:
        return []
    elif task in ["LaMP_5", 'LaMP_t_5']:
        return []
    elif task in ["LaMP_6", 'LaMP_t_6']:
        return []
    elif task in ["LaMP_7", 'LaMP_t_7']:
        return []


def get_suffix(task):
    # detail prompt from sbp
    suffix = {'LaMP_3': """Based on the review, please analyze its sentiment
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
    """}
    if suffix.get(task):
        return suffix[task]
    else:
        print(f' No implemented suffix for task {task}')


def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        # model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True,)
        model_inputs = tokenizer(
            inputs,  # 输入
            text_target=targets,  # 输出
            max_length=max_length,
            truncation=True,
            padding=True
        )
        return model_inputs

    return preprocess_dataset


def create_preprocessor4wh(tokenizer, bge_tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        profile_mid = [example for example in examples['profile']]
        profiles = []

        for x in profile_mid:
            profiles.extend(x)

        model_inputs = tokenizer(inputs,  # 输入
                                 text_target=targets,  # 输出
                                 max_length=max_length, truncation=True, padding=True)
        bge_query = bge_tokenizer(inputs, return_tensors='pt',
                                  padding='max_length', truncation=True,
                                  max_length=512)
        bge_history = bge_tokenizer(profiles, return_tensors='pt',
                                    padding='max_length', truncation=True,
                                    max_length=512)

        model_inputs['bge_query_ids'] = bge_query['input_ids']
        model_inputs['bge_query_mask'] = bge_query['attention_mask']

        bs = bge_query['attention_mask'].shape[0]
        model_inputs['bge_history_ids'] = bge_history['input_ids'].reshape(bs, -1)
        model_inputs['bge_history_mask'] = bge_history['attention_mask'].reshape(bs, -1)

        return model_inputs

    return preprocess_dataset


def create_preprocessor_scores(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        # model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True,)
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True, padding=True)
        model_inputs['id_1'] = examples['id_1']
        model_inputs['id_2'] = examples['id_2']
        return model_inputs

    return preprocess_dataset


def create_preprocessor_scores_seq(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        model_inputs['id'] = examples['id']
        return model_inputs

    return preprocess_dataset


def task_template(task, input_data):
    # default task template
    task_templates = {
        "LaMP_1": "For an author who has written the paper with the title \"{title}\", which reference is related? Just answer with [1] or [2] without explanation.\n[1]: \"{ref1}\"\n[2]: \"{ref2}\"",
        "LaMP_2": "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation.\ntags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, …] description: {movie}",
        "LaMP_3": "What is the score of the following review on a scale of 1 to 5? Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",
        "LaMP_4": "Generate a headline for the following article: {article}",
        "LaMP_5": "Generate a title for the following abstract of a paper: {abstract}",
        "LaMP_6": "Generate a subject for the following email: {email}",
        "LaMP_7": "Paraphrase the following tweet without any explanation before or after it: {tweet}",
    }
    if task in task_templates:
        template = task_templates[task].format(**input_data)
        return template
    else:
        raise ValueError("Invalid task type. Task should be one of LaMP-1 to LaMP-7.")


def sample_profiles_by_score(profiles, n):
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


def cus_input(task, **kwargs):
    if task == 'LaMP_3':
        # TODO  添加 {task prompt , answer prompt , pivot sample, sample num, user profile, score distribution}
        lamp_3_template = Template("""
            **User Profile**
            {{profile}}
            **History Sample**
            {{sample}}
            **Task**
            {{input}}
            **Task Instruction**
            Follow your previous rating habits and these instructions:
            • If you feel satisfied with this product or have concerns but it’s good overall, it should be rated 5.
            • If you feel good about this product but notice some issues, it should be rated as 4.
            • If you feel OK but have concerns, it should be rated as 3.
            • If you feel unsatisfied with this product but it’s acceptable for some reason, it should be rated as 2.
            • If you feel completely disappointed or upset, it should be rated 1.
            **User Score Distribution**
            {{dist}}
            """.replace('\t', ''))
        # return lamp_3_template.render(profile=profile, sample=sample, input=input, dist=dist)
        return lamp_3_template.render(**kwargs)
    elif task == 'LaMP_7':
        """
        **User Profile**
        {{profile}}
        **History Sample**
        {{sample}}
        """
        lamp_7_template = Template(
            """
             **User Profile**
            {{profile}}
            **History Sample**
            {{sample}}
            **Task***
            {{input}}
            """.replace('\t', ''))
        # return lamp_3_template.render(profile=profile, sample=sample, input=input, dist=dist)
        return lamp_7_template.render(**kwargs)


def load_data4plora(model_args, data_args, training_args, train=True, with_reason=True):
    """
    https://hugging-face.cn/docs/trl/sft_trainer
    trl SFT 对话格式
    {"messages": [
     {"role": "system", "content": "You are helpful"},
     {"role": "user", "content": "What's the capital of France?"},
     {"role": "assistant", "content": "..."}
     ]}
    指令格式
    {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

    TODO 从history采样训练数据， 暂时 val_size * 5
    """

    personal_type = model_args.personal_type  # uid ,profile

    dataset_text_field = training_args.dataset_text_field
    task = data_args.dataset_name
    retriever = model_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    # train_data_path = f'{abs_dir}/rank/{task}/train_questions_rank_{retriever}_merge.json'
    validation_data_path = f'{abs_dir}/rank/{task}/dev_questions_rank_{retriever}_merge.json'

    with open(validation_data_path) as file:
        data = json.load(file)

    def cut_max(x):
        return ' '.join(x.split(' ')[:training_args.max_seq_length - 200])

    idmap = {x['id']: i for i, x in enumerate(data)}
    score_dists = [cal_dist(x['profile']) for x in data]

    # TODO
    profile_method = 'SBP'  # SBP, SBP_div
    profile_path = f'{abs_dir}/profile/{task}/dev_{profile_method}_' \
                   f'Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
    with jsonlines.open(profile_path, 'r') as f:
        profiles = list(f)  # input output
        profiles = [x['output'] for x in profiles]

    # TODO 给 plora 数据生成 reason    -> p_train
    reason_dir = f'../output/{task}/STaR'
    with jsonlines.open(f"{reason_dir}/{task}_with_reason&profile_p_train.jsonl", 'r') as f:
        train_reasons = list(f)
    with jsonlines.open(f"{reason_dir}/{task}_with_reason&profile_p_train.jsonl", 'r') as f:
        eval_reasons = list(f)

    profile_embs = None
    if personal_type == 'profile':
        profile_tensor_path = f'{abs_dir}/profile/{task}/dev_SBP_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.pt'
        if os.path.exists(profile_tensor_path):
            profile_embs = torch.load(profile_tensor_path)
        else:
            profile_embs = []
            # 加载 嵌入模型
            device = 'cuda:0'
            model_path = '/data/hf/BAAI/bge-base-en-v1.5'
            tokenizer = AutoTokenizer.from_pretrained('/data/hf/BAAI/bge-base-en-v1.5')
            profile_encoder = AutoModel.from_pretrained(model_path).to(device)
            profile_encoder.eval()
            bs = 16
            with torch.no_grad():
                for i in range(int(len(profiles) / bs) + 1):
                    ps = profiles[i * bs:bs * (i + 1)]
                    profile_input_ids = tokenizer(ps, return_tensors='pt', padding='max_length',
                                                  truncation=True, max_length=512).to(device)
                    # cls pooling  ref:https://huggingface.co/BAAI/bge-large-zh
                    emb = torch.nn.functional.normalize(profile_encoder(**profile_input_ids)[0][:, 0])  # (bs 768)
                    profile_embs.append(emb)

            profile_embs = torch.cat(profile_embs, dim=0)
            torch.save(profile_embs, profile_tensor_path)
            profile_embs = torch.load(profile_tensor_path)

    # TODO
    model_args.user_num = len(idmap)  # 设置 user embedding size
    eval_data = [{
        'p': idmap[x['id']] if personal_type == 'uid' else profile_embs[idmap[x['id']], :],
        'input': cus_input(task,
                           profile=profiles[i],
                           sample=x['profile'][:1],  # random.randint(0, len(x['profile']) - 1)
                           input=cut_max(x['input']),
                           dist=score_dists[i]),
        'output': eval_reasons[i]['output'] if with_reason else x['output'],
    } for i, x in enumerate(data)]

    train_data = []
    for i, item in enumerate(data):
        history = sample_profiles_by_score(item['profile'], 5)  # 采样历史数据
        temp = []
        uid = idmap[item['id']]
        for j, x in enumerate(history):
            x['text'] = cut_max(x['text'])
            # 历史数据需要额外添加task模板
            temp.append({
                'p': uid if personal_type == 'uid' else profile_embs[uid, :],
                'input': cus_input(task,
                                   profile=profiles[i],
                                   sample=history[min(j + 1, len(history) - 1)],
                                   input=cut_max(task_template(task, x)),
                                   dist=score_dists[i],
                                   ),
                'output': x['score'] if not with_reason else train_reasons[5 * i + j]['output'],
            })
        train_data.extend(temp)

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(train_data).shuffle()
    my_datasets['dev'] = HFDataset.from_list(eval_data).shuffle()

    def apply_chat_format(sample):
        if train is False:
            chat_text = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample['input']},
            ]
        else:  # train with input and answer
            chat_text = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample['input']},
                {"role": "assistant", "content": sample['output']}
            ]
        return {dataset_text_field: chat_text}

    my_datasets = my_datasets.map(
        apply_chat_format,
    )
    return my_datasets


def load_data4lora(model_args, data_args, training_args, train=True, apply_chat_template=True):
    """
    https://hugging-face.cn/docs/trl/sft_trainer
    trl SFT 对话格式
    {"messages": [
     {"role": "system", "content": "You are helpful"},
     {"role": "user", "content": "What's the capital of France?"},
     {"role": "assistant", "content": "..."}
     ]}
    指令格式
    {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    """

    dataset_text_field = training_args.dataset_text_field
    task = data_args.dataset_name
    retriever = 'contriever'  # model_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'

    def cut_max(x):
        return ' '.join(x.split(' ')[:training_args.max_seq_length - 200])

    def make_input(data_path, profile_path):
        with open(data_path) as file:
            data = json.load(file)

        with jsonlines.open(profile_path, 'r') as f:
            profiles = list(f)  # input output
            profiles = [x['output'] for x in profiles]
            profiles = [x.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1] for x in profiles]

        score_dists = [cal_dist(x['profile']) for x in data]

        ret = [{
            'input': cus_input(
                task,
                profile=profiles[i],
                sample=x['profile'][:1],  # random.randint(0, len(x['profile']) - 1)
                input=cut_max(x['input']),
                dist=score_dists[i],  # lamp_3 only
            ),
            'output': x['output'],
        } for i, x in enumerate(data)]
        return ret

    train_data_path = f'{abs_dir}/rank/{task}/train_questions_rank_{retriever}_merge.json'
    eval_data_path = f'{abs_dir}/rank/{task}/dev_questions_rank_{retriever}_merge.json'
    profile_method = 'SBP'  # SBP, SBP_div
    train_profile_path = f'{abs_dir}/profile/{task}/train_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
    eval_profile_path = f'{abs_dir}/profile/{task}/dev_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'

    train_data = make_input(train_data_path, train_profile_path)
    eval_data = make_input(eval_data_path, eval_profile_path)

    my_datasets = DatasetDict()

    my_datasets['train'] = HFDataset.from_list(train_data).shuffle()
    my_datasets['dev'] = HFDataset.from_list(eval_data).shuffle()

    def apply_chat_format(sample):
        if not train:  # infer mode
            chat_text = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample['input']},
            ]
        else:
            chat_text = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample['input']},
                {"role": "assistant", "content": sample['output']}
            ]
        return {dataset_text_field: chat_text}

    if apply_chat_template:  # vllm no need
        my_datasets = my_datasets.map(apply_chat_format, )
    return my_datasets


if __name__ == '__main__':
    """
    
    # /data/hf/meta-llama/Llama-3.2-1B-Instruct
    # /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
    # /data/hf/google/flan-t5-base
    
    --method SFT
    --task LaMP_7
    --model /data/hf/meta-llama/Llama-3.2-1B-Instruct
    --retriever contriever
    --num_retrieved 4
    --use_profile
    --is_ranked
    
    """
    from LaMP.LaMP.args import load_args

    # from PLoRA.cfgs.new_config import load_args

    args = load_args()
    # dataset = load_data(args)
    dataset = load_data4plora(args)
    print()
