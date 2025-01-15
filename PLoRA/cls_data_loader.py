import os
import random

from tqdm import tqdm

import json
import jsonlines

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModel

from datasets import DatasetDict
from datasets import Dataset as HFDataset

from data.datasets import (
    get_all_labels,
    GeneralSeq2SeqDataset,
    create_preprocessor,
    convert_to_hf_dataset
)
from prompts.prompts import create_prompt_generator

from jinja2 import Template
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

MSL = 512
idmap = dict()


def sample_data_raw(data, task, sample_num):
    # bianli profile
    extra_sample = []
    for x in data:
        sample_index = random.sample(range(len(x['profile'])), sample_num)
        for si in sample_index:
            si = min(max(si, 1), len(x['profile'] - 2))  # 对应最小 profile_num
            item = x['profile'][si]
            extra_sample.append({
                'id': x['id'],
                'input': apply_task_template(task, item)[0],  # TODO
                'profile': x['profile'][si + 1:],  # profile 降序排列
                'output': apply_task_template(task, item)[1]  # TODO
            })
    data = data + extra_sample
    random.shuffle(data)
    return data


def sample_data(data, profile_embs, task, sample_n=2):
    # n = len(data)
    # K = int(n / (sample_n + 1))  # 采样聚类中心数量
    #
    # # pip install umap-learn
    # # Step 1: 使用 UMAP 进行降维  Uniform Manifold Approximation and Projection
    # umap_model = umap.UMAP(n_components=10)  # 降到二维 /10
    # reduced_data = umap_model.fit_transform(profile_embs.cpu())
    # # Step 2: 使用 K-means 聚类
    # kmeans = KMeans(n_clusters=K)
    # labels = kmeans.fit_predict(reduced_data)
    # centroids = kmeans.cluster_centers_  # # 聚类中心
    # # 计算每个点与每个聚类中心的距离
    # distances = np.linalg.norm(reduced_data[:, np.newaxis] - centroids, axis=2)
    # # 对于每个聚类中心，获取最近的点的索引
    # closest_indices = np.argmin(distances, axis=0)
    #
    # # TODO 采样数据
    # data = [data[i] for i in closest_indices]

    # TODO filter & boost
    # bianli profile
    extra_sample = []
    for x in data:
        sample_index = random.sample(range(len(x['profile'])), sample_n)
        for si in sample_index:
            si = min(max(si, 1), len(x['profile']) - 2)  # 对应最小 profile_num
            item = x['profile'][si]
            extra_sample.append({
                'id': x['id'],
                'input': apply_task_template(task, item)[0],  # TODO
                'profile': x['profile'][si + 1:],  # profile 降序排列
                'output': apply_task_template(task, item)[1]  # TODO
            })
    data = data + extra_sample
    # random.shuffle(data)
    return data


def trunc_input(tokenizer, x):
    max_length = 312
    # 使用tokenizer对文本进行编码，并截断文本
    encoded_input = tokenizer(x, max_length=max_length, truncation=True, padding=False,
                              return_tensors="pt")
    truncated_text = tokenizer.decode(encoded_input['input_ids'].squeeze())
    return truncated_text


def apply_task_template(task, profile):
    input_text = {
        "LaMP_1": "For an author who has written the paper with the title \"{title}\", "
                  "which reference is related? Just answer with [1] or [2] without explanation."
                  "\n[1]: \"{ref1}\"\n[2]: \"{ref2}\"",

        "LaMP_2": "Which tag does this movie relate to among the following tags? "
                  "Just answer with the tag name without further explanation."
                  "\ntags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, …] "
                  "description: {movie}",

        "LaMP_3": "What is the score of the following review on a scale of 1 to 5? "
                  "Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",

        "LaMP_4": "Generate a headline for the following article: {text}",  # no article

        "LaMP_5": "Generate a title for the following abstract of a paper: {abstract}",

        "LaMP_6": "Generate a subject for the following email: {email}",

        "LaMP_7": "Paraphrase the following tweet without any explanation before or after it: {tweet}",

        # TODO
        "IMDB-B": "What is the star of the following movie reviews, "
                  "recommendation ratings for each review ranging from 1-10 stars."
                  "Just answer with 1-10 without further explanation.\nreview: {text}",
        "YELP-B": "What is the star of the following restaurant reviews, "
                  "Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",
        "GDRD-B": "What is the star of the following book reviews, "
                  "Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",
        "PPR-B": "What is the star of the following food reviews, "
                 "Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",

    }[task].format(**profile)
    output = {
        # 'LaMP_1': "{abstract}",
        # 'LaMP_2': "{tag}",
        'LaMP_3': "{score}",
        'LaMP_4': "{title}",
        'LaMP_5': "{title}",
        # 'LaMP_6': "{subject}",
        # 'LaMP_7': "{tweet}",
        # TODO
        'IMDB-B': '{score}',
        'YELP-B': '{score}',
        'GDRD-B': '{score}',
        'PPR-B': '{score}',
    }[task].format(**profile)
    return (input_text, output)


def apply_history_template(task, **kwargs):
    if task in ['LaMP_1']:
        return Template("Title: {{title}} Abstract: {{abstract}}\n").render(**kwargs)
    elif task in ['LaMP_2']:
        return Template("Description：{{movie}} Tag {{tag}}\n").render(**kwargs)
    elif task in ['LaMP_3']:
        return Template("Text: {{text}} Score: {{score}}\n").render(**kwargs)
    elif task in ['LaMP_4']:
        return Template("Title: {{title}} text: {{article}}\n").render(**kwargs)
    elif task in ['LaMP_5']:
        return Template("Title: {{title}} text: {{abstract}}\n").render(**kwargs)
    elif task in ["LaMP_7"]:
        return Template("Text: {{tweet}}\n").render(**kwargs)
    # TODO
    elif task in ["IMDB-B", 'YELP-B', 'GDRD-B', 'PPR-B']:
        return Template("Text: {{text}} Score: {{score}}\n").render(**kwargs)


def load_profile(abs_dir, task, split, retriever):
    profile_method = 'SBP'  # SBP, SBP_div
    profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
    profile_tensor_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.pt'

    with jsonlines.open(profile_path, 'r') as f:
        profiles = list(f)  # input output
        profiles = [x['output'] for x in profiles]

    device = 'cuda:0'
    if os.path.exists(profile_tensor_path):
        profile_embs = torch.load(profile_tensor_path, map_location=torch.device(device))
    else:
        profile_embs = []
        # 加载 嵌入模型
        model_path = '/data/hf/BAAI/bge-base-en-v1.5'
        emb_tokenizer = AutoTokenizer.from_pretrained('/data/hf/BAAI/bge-base-en-v1.5')
        profile_encoder = AutoModel.from_pretrained(model_path).to(device)
        profile_encoder.eval()
        bs = 16
        with torch.no_grad():
            for i in range(int(len(profiles) / bs) + 1):
                ps = profiles[i * bs:bs * (i + 1)]
                if len(ps) == 0:
                    continue
                profile_input = emb_tokenizer(ps, return_tensors='pt', padding='max_length',
                                              truncation=True, max_length=512).to(device)
                # cls pooling  ref:https://huggingface.co/BAAI/bge-large-zh
                emb = torch.nn.functional.normalize(profile_encoder(**profile_input)[0][:, 0])  # (bs 768)
                profile_embs.append(emb)

        profile_embs = torch.cat(profile_embs, dim=0)
        torch.save(profile_embs, profile_tensor_path)
        profile_embs = torch.load(profile_tensor_path, map_location=torch.device(device))

    return profiles, profile_embs


def load_dataset4plora_profile(model_args, data_args, training_args, tokenizer, tokenize=True):
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    msl = MSL  # training_args.max_seq_length
    history_num = data_args.num_retrieved

    prompt_generator, _ = create_prompt_generator(history_num, retriever,
                                                  is_ranked=True,
                                                  max_length=msl,
                                                  tokenizer=tokenizer)

    def make_data(split, sn=0):
        assert split in ['train', 'dev', 'test']

        if personal_type == 'profile':
            profiles, profile_embs = load_profile(abs_dir, task, split, retriever)

        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)  # [:64]

        idmap = {x['id']: i for i, x in enumerate(data)}

        processed_data = []
        for i, x in tqdm(enumerate(data)):

            if split == 'train':
                history_shift = 0
            else:
                history_shift = min(0, len(x['profile']) - history_num)

            x['input'] = trunc_input(tokenizer, x['input'])
            histories = x['profile'][history_shift:]
            # random.shuffle(histories)

            prompt_types = data_args.prompt.split(',')  # default='history,prompt,query'
            if 'profile' in prompt_types and 'history' in prompt_types:  # profile and history
                input_text = prompt_generator(
                    f"**User Profile**\n{profiles[i]}\n**Task Description**\n{x['input']}",
                    histories,
                    task)
            elif 'profile' in prompt_types and 'history' not in prompt_types:  # profile only
                input_text = f"**User Profile**\n{profiles[i]}\n**Task Description**\n{x['input']}"
            elif 'history' in prompt_types and data_args.num_retrieved > 0:
                input_text = prompt_generator(
                    x['input'],
                    histories,
                    '**Task Description**\n' + task)
            else:  # query only
                input_text = '**Task Description**\n' + x['input']

            input_text = input_text.replace('<|begin_of_text|>', '')

            pe = profile_embs[i, :] if task in ['IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B'] \
                else profile_embs[idmap[x['id']], :]

            processed_data.append({
                'p': pe,
                'source': input_text,
                'target': str(x['output']),
            })
        # raw id ,source target
        # TODO sum([x['source'][19]== x['target']  for x in processed_data])

        if sn == 0:
            return processed_data  # [:64]  # for test
        else:
            return processed_data[:sn]

    my_datasets = DatasetDict()  # TODO 注意 和示例答案重合度 85%
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    if task in ['IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
        my_datasets['test'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    if tokenize:  # 2 input_ids & label
        my_datasets = my_datasets.map(
            create_preprocessor(tokenizer=tokenizer, max_length=512),
            batched=True,
            remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_datasets


def filter_sim_label(x, task):
    target_label = x['output']
    histories = []
    for his in x['profile']:
        d = {
            'LaMP_3': "score",
            'LaMP_4': "title",
            'LaMP_5': "title",
        }
        if his[d[task]] == target_label:
            histories.append(his)
    if len(histories) > 0:
        return histories
    else:
        return x['profile']


def load_dataset4cls(model_args, data_args, training_args, tokenizer, tokenize=True):
    # profile
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    msl = training_args.max_seq_length  # MSL
    history_num = data_args.num_retrieved

    prompt_generator, _ = create_prompt_generator(history_num, retriever,
                                                  is_ranked=True,
                                                  max_length=msl,
                                                  tokenizer=tokenizer)

    def make_data(split):
        assert split in ['train', 'dev', 'test']

        profiles, profile_embs = load_profile(abs_dir, task, split, retriever)
        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)  #[:64]

        processed_data = []
        for i, x in tqdm(enumerate(data)):
            x['input'] = trunc_input(tokenizer, x['input'])
            histories = x['profile']

            prompt_types = data_args.prompt.split(',')  # default='history,prompt,query'
            if 'profile' in prompt_types and 'history' in prompt_types:  # profile and history
                input_text = prompt_generator(
                    f"**User Profile**\n{profiles[i]}\n**Task Description**\n{x['input']}",
                    histories,
                    task)
            elif 'profile' in prompt_types and 'history' not in prompt_types:  # profile only
                input_text = f"**User Profile**\n{profiles[i]}\n**Task Description**\n{x['input']}"
            elif 'history' in prompt_types and data_args.num_retrieved > 0:
                input_text = prompt_generator(
                    '**Task Description**\n' + x['input'],
                    histories, task)
            else:  # query only
                input_text = '**Task Description**\n' + x['input']

            input_text = input_text.replace('<|begin_of_text|>', '')
            input_text = input_text.replace('</s>', '')
            processed_data.append({
                'p': profile_embs[i, :],
                'source': input_text,
                'labels': int(x['output']) - 1,
            })
        return processed_data

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    if task in ['IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
        my_datasets['test'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    def preprocess_function(examples):
        return tokenizer(examples['source'], truncation=True, padding="longest", )

    if tokenize:  # 2 input_ids & label
        my_datasets = my_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=['source', ])  # p, input_ids, attention_mask, labels
    return my_datasets
