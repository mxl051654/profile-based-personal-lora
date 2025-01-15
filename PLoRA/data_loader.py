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
    if task in ['LaMP_3', 'LaMP_4', 'LaMP_5']:
        extra_sample = []
        ids = [i for i in range(len(data))]
        for i, x in enumerate(data):
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
            ids.extend([i] * sample_n)
        data = data + extra_sample
    else:
        # da with random history
        extra_sample = []
        ids = [i for i in range(len(data))]
        for i, x in enumerate(data):
            sample_index = random.sample(range(len(x['profile'])), sample_n)
            for si in sample_index:
                his = x['profile']
                random.shuffle(his)
                extra_sample.append({
                    'id': x['id'],
                    'input': x['input'],  # TODO
                    'profile': his,  # profile 降序排列
                    'output': x['output']  # TODO
                })
            ids.extend([i] * sample_n)
        data = data + extra_sample
    return data, ids


def trunc_input(tokenizer, x):
    max_length = 312
    # 使用tokenizer对文本进行编码，并截断文本
    encoded_input = tokenizer(x, max_length=max_length, truncation=True, padding=False,
                              return_tensors="pt")
    truncated_text = tokenizer.decode(encoded_input['input_ids'].squeeze())
    return truncated_text


def apply_task_template(task, profile):
    input_text = {
        # "LaMP_1": "For an author who has written the paper with the title \"{title}\", "
        #           "which reference is related? Just answer with [1] or [2] without explanation."
        #           "\n[1]: \"{ref1}\"\n[2]: \"{ref2}\"",
        "LaMP_1": "Author's past paper with title: {title} \n and abstract: {abstract}",

        "LaMP_2": "Which tag does this movie relate to among the following tags? "
                  "Just answer with the tag name without further explanation."
                  "\ntags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, …] "
                  "description: {movie}",

        "LaMP_3": "What is the score of the following review on a scale of 1 to 5? "
                  "Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",

        "LaMP_4": "Generate a headline for the following article: {text}",  # no article

        "LaMP_5": "Generate a title for the following abstract of a paper: {abstract}",
        "LaMP_6": "Generate a subject for the following email: {email}",

        # "LaMP_7": "Author's past tweet: {tweet}",
        "LaMP_7": "Author's past tweet: {text}",
        # TODO
        "LaMP_t_1": "Author's past paper with title: {title} \n and abstract: {abstract}",
        "LaMP_t_3": "What is the score of the following review on a scale of 1 to 5? "
                    "Just answer with 1, 2, 3, 4, or 5 without further explanation.\nreview: {text}",
        "LaMP_t_4": "Generate a headline for the following article: {text}",  # no article
        "LaMP_t_5": "Generate a title for the following abstract of a paper: {abstract}",
        "LaMP_t_7": "Author's past tweet: {text}",
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
        'LaMP_1': "None",
        # 'LaMP_2': "{tag}",
        'LaMP_3': "{score}",
        'LaMP_4': "{title}",
        'LaMP_5': "{title}",
        # 'LaMP_6': "{subject}",
        'LaMP_7': "None",
        # TDDO:
        'LaMP_t_1': "None",
        'LaMP_t_3': "{score}",
        'LaMP_t_4': "{title}",
        'LaMP_t_5': "{title}",
        'LaMP_t_7': "None",
        # TODO
        'IMDB-B': '{score}',
        'YELP-B': '{score}',
        'GDRD-B': '{score}',
        'PPR-B': '{score}',
    }[task].format(**profile)
    return (input_text, output)


def apply_history_template(task, **kwargs):
    if task in ['LaMP_1', 'LaMP_t_1']:
        return Template("Title: {{title}} Abstract: {{abstract}}\n").render(**kwargs)
    elif task in ['LaMP_2', 'LaMP_t_2']:
        return Template("Description：{{movie}} Tag {{tag}}\n").render(**kwargs)
    elif task in ['LaMP_3', 'LaMP_t_3']:
        return Template("Text: {{text}} Score: {{score}}\n").render(**kwargs)
    elif task in ['LaMP_4', 'LaMP_t_4']:
        return Template("Title: {{title}} text: {{article}}\n").render(**kwargs)
    elif task in ['LaMP_5', 'LaMP_t_5']:
        return Template("Title: {{title}} text: {{abstract}}\n").render(**kwargs)
    elif task in ["LaMP_7", 'LaMP_t_7']:
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


def load_dataset4plora_id(model_args, data_args, training_args, tokenizer):
    # TODO 注意 plora_id 使用val set 的 history进行训练
    # TODO 后续 考虑 时间划分 [需要预先分配 Embedding layer size]
    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    msl = MSL  # training_args.max_seq_length
    prompt_generator, _ = create_prompt_generator(data_args.num_retrieved, retriever,
                                                  is_ranked=True,
                                                  max_length=msl,
                                                  tokenizer=tokenizer)
    his_num = data_args.num_retrieved  # 2

    def make_data(split, sn=0):
        assert split in ['train', 'dev', 'test']

        global idmap

        if split in 'train':
            # 从 evaluate history 采样
            data_path = f'{abs_dir}/rank/{task}/train_questions_rank_{retriever}_merge.json'
            with open(data_path) as f:
                temp_train_data = json.load(f)

            total_sample_num = len(temp_train_data)  # 对齐训练数据量

            data_path = f'{abs_dir}/rank/{task}/dev_questions_rank_{retriever}_merge.json'
            with open(data_path) as f:
                data = json.load(f)

            sample_num = int(total_sample_num / len(data))

            if task in ['IMDB-B', 'YELP-B']:
                uids = set([x['id'] for x in data])
                idmap = {uid: i for i, uid in enumerate(uids)}
            else:
                idmap = {x['id']: i for i, x in enumerate(data)}

            model_args.user_size = len(idmap)

            train_data = []
            for x in data:
                cur_profiles = x['profile']

                sample_indices = random.sample(list(range(his_num + 1, len(cur_profiles))),
                                               min(sample_num, len(cur_profiles) - his_num - 1))

                pairs = [apply_task_template(task, cur_profiles[i]) for i in sample_indices]

                profiles = [cur_profiles[i - his_num:i] for i in sample_indices]
                for idx, pair in enumerate(pairs):
                    train_data.append({
                        'p': idmap[x['id']],
                        'source': prompt_generator(pair[0], profiles[idx], task) if data_args.num_retrieved > 0
                        else pair[0],
                        'target': str(pair[1]),
                    })
            return train_data
        else:
            data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
            with open(data_path) as f:
                data = json.load(f)

            processed_data = [{
                'p': idmap[x['id']],
                'source': prompt_generator(x['input'], x['profile'], task) if data_args.num_retrieved > 0
                else x['input'],
                'target': str(x['output']),
            } for i, x in enumerate(data)]  # raw id ,source target

            return processed_data

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    if task in ['IMDB-B', 'YELP-B']:
        my_datasets['dev'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    my_dataset = my_datasets.map(
        create_preprocessor(tokenizer=tokenizer, max_length=msl),
        batched=True,
        remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_dataset


def load_dataset4pplug_history(model_args, data_args, training_args, tokenizer):
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    msl = MSL  # training_args.max_seq_length

    prompt_generator, _ = create_prompt_generator(data_args.num_retrieved, retriever,
                                                  is_ranked=True,
                                                  max_length=msl,
                                                  tokenizer=tokenizer)

    def make_data(split, sn=0):
        assert split in ['train', 'dev', 'test']

        his_num = data_args.num_retrieved  # 2
        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)
        inputs = [x['input'] for x in data]

        def sample_history(his, tn):
            if len(his) >= tn:
                return his[:tn]
            else:
                return random.choices(his, k=tn)

        histories = [sample_history(x['profile'], his_num) for x in data]
        history_tensor_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge_history.pt'

        if personal_type == 'history':
            if os.path.exists(history_tensor_path):
                print('load history f')
                histories_embs = torch.load(history_tensor_path)
            else:
                device = 'cuda:0'
                model_path = '/data/hf/BAAI/bge-base-en-v1.5'
                emb_tokenizer = AutoTokenizer.from_pretrained('/data/hf/BAAI/bge-base-en-v1.5')
                encoder = AutoModel.from_pretrained(model_path).to(device)
                encoder.eval()
                history_embs = []
                with torch.no_grad():
                    for ui, his in zip(inputs, histories):
                        # 拼接 history info
                        his = [apply_history_template(task, **hi) for hi in his]
                        his_input = emb_tokenizer(his, return_tensors='pt', padding='max_length',
                                                  truncation=True, max_length=512).to(device)
                        his_emb = torch.nn.functional.normalize(encoder(**his_input)[0][:, 0])  # (bs 768)
                        #
                        u_input = emb_tokenizer(ui, return_tensors='pt', padding='max_length',
                                                truncation=True, max_length=512).to(device)
                        u_emb = torch.nn.functional.normalize(encoder(**u_input)[0][:, 0])  # (bs 768)

                        history_embs.append(torch.cat([his_emb, u_emb], dim=0))

                histories_embs = torch.stack(history_embs, dim=0)  # data_num , his_num + 1, 768
                torch.save(histories_embs, history_tensor_path)
                histories_embs = torch.load(history_tensor_path)

        idmap = {x['id']: i for i, x in enumerate(data)}

        processed_data = [{
            'p': histories_embs[idmap[x['id']], :],
            'source': prompt_generator(x['input'], x['profile'], task) if data_args.num_retrieved > 0 else x['input'],
            'target': str(x['output']),
        } for i, x in enumerate(data)]  # raw id ,source target
        if sn == 0:
            return processed_data  # [:64]  # for test
        else:
            return processed_data[:sn]

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    if task in ['IMDB-B', 'YELP-B']:
        my_datasets['dev'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    my_dataset = my_datasets.map(
        create_preprocessor(tokenizer=tokenizer, max_length=msl),
        batched=True,
        remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_dataset


def load_dataset4plora_profile(model_args, data_args, training_args, tokenizer, tokenize=True):
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    # msl = MSL  # training_args.max_seq_length
    msl = training_args.max_seq_length  # TODO MSL
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

        # idmap = {x['id']: i for i, x in enumerate(data)}

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
                    '**Task Description**\n' + x['input'],
                    histories, task)
            else:  # query only
                input_text = '**Task Description**\n' + x['input']

            input_text = input_text.replace('<|begin_of_text|>', '')

            processed_data.append({
                'p': profile_embs[i, :],
                'source': input_text,
                'target': str(x['output']),
            })
        # raw id ,source target
        # TODO
        #  63% for PPR-B  [0, 80, 320, 1520, 5680]
        #  34% GDRD-B  [238, 626, 1198, 1829, 1033]
        #  33% YELP-B  [2946, 4035, 6692, 14629, 13882]
        #  15% IMDB-B  [485, 350, 337, 201, 708, 793, 1260, 1815, 1009, 1017]
        # 标签分布 [sum([str(i+1)== x['target']  for x in processed_data]) for i in range(5)]
        # 提示答案相似度 sum([x['source'][19]== x['target']  for x in processed_data])/len(processed_data)

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


def load_dataset4plora_moe(model_args, data_args, training_args, tokenizer):
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
        profile_method = 'SBP'  # SBP, SBP_div
        train_cluster_path = f'{abs_dir}/profile/{task}/train_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024_cluster.npy'

        if personal_type == 'profile':
            profiles, profile_embs = load_profile(abs_dir, task, split, retriever)
        profile_embs = profile_embs.cpu().numpy()

        # TODO cal weight for moe exp
        if split == 'train':
            # pip install umap-learn
            # Step 1: 使用 UMAP 进行降维  Uniform Manifold Approximation and Projection
            umap_model = umap.UMAP(n_components=10)  # 降到二维
            reduced_data = umap_model.fit_transform(profile_embs)
            if not os.path.exists(train_cluster_path):
                # Step 2: 使用 K-means 聚类
                K = model_args.expert_num
                kmeans = KMeans(n_clusters=K)
                labels = kmeans.fit_predict(reduced_data)
                train_clusters = kmeans.cluster_centers_  # # 聚类中心
                np.save(train_cluster_path, train_clusters)
            else:
                train_clusters = np.load(train_cluster_path)
        else:
            umap_model = umap.UMAP(n_components=10)  # 降到二维
            reduced_data = umap_model.fit_transform(np.array(profile_embs))
            train_clusters = np.load(train_cluster_path)
        # 计算每个点到每个聚类中心的欧氏距离
        distances = cdist(reduced_data, train_clusters, 'euclidean')  # 形状: [n_samples, K]
        # 归一化距离向量（按行归一化，使得每个样本的距离之和为 1）
        moe_weights = torch.tensor(normalize(distances, norm='l1', axis=1))

        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)

        idmap = {x['id']: i for i, x in enumerate(data)}

        processed_data = []
        for i, x in enumerate(data):
            # TODO shift time type
            # TODO label shift (刻意展示和测试不同分布的数据)
            if split == 'train':
                history_shift = 0
            else:
                history_shift = min(0, len(x['profile']) - history_num)

            processed_data.append({
                # TODO
                # 'p': moe_weights[idmap[x['id']], :],  # 基于个性化信息的Router
                'p': profile_embs[idmap[x['id']], :],
                'source': prompt_generator(x['input'],
                                           # random.sample(x['profile'], k=min(10,len(x['profile']))),
                                           x['profile'][history_shift:],
                                           # x['profile'],
                                           task) if data_args.num_retrieved > 0 else x['input'],
                'target': str(x['output']),
            })
        # raw id ,source target
        if sn == 0:
            return processed_data  # [:64]  # for test
        else:
            return processed_data[:sn]

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train', sn=0)).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    if task in ['IMDB-B', 'YELP-B']:
        my_datasets['dev'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    my_dataset = my_datasets.map(
        create_preprocessor(tokenizer=tokenizer, max_length=msl),
        batched=True,
        remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_dataset


def load_dataset4plora_profile_few_shots(model_args, data_args, training_args, tokenizer, tokenize=True):
    """

    """
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'

    msl = MSL  # training_args.max_seq_length
    history_num = data_args.num_retrieved

    prompt_generator, _ = create_prompt_generator(
        history_num, retriever,
        is_ranked=True,
        max_length=msl,
        tokenizer=tokenizer)

    def make_data(split, sn=0):
        assert split in ['train', 'dev', 'test']
        profile_method = 'SBP'  # SBP, SBP_div
        profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
        profile_tensor_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.pt'

        with jsonlines.open(profile_path, 'r') as f:
            profiles = list(f)  # input output
            profiles = [x['output'] for x in profiles]

        profile_embs = None  # profile emb
        if personal_type == 'profile':
            if os.path.exists(profile_tensor_path):
                profile_embs = torch.load(profile_tensor_path)
            else:
                profile_embs = []
                # 加载 嵌入模型
                device = 'cuda:0'
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
                profile_embs = torch.load(profile_tensor_path)

        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)

        idmap = {x['id']: i for i, x in enumerate(data)}

        if split == 'train':
            # TODO sample data

            processed_data = []
            for i, x in enumerate(data):

                if split == 'train':
                    history_shift = 0
                else:
                    history_shift = min(0, len(x['profile']) - history_num)

                # TODO
                x['input'] = trunc_input(tokenizer, x['input'])
                #
                prompt_types = data_args.prompt.split(',')  # default='history,prompt,query'
                if 'profile' in prompt_types and 'history' in prompt_types:  # TODO profile and history
                    input_text = prompt_generator(f"**user profile**{profiles[i]}\n**task**{x['input']}",
                                                  x['profile'][history_shift:], task)
                elif 'profile' in prompt_types and 'history' not in prompt_types:  # TODO profile only
                    input_text = f"**user profile**{profiles[i]}\n**task**{x['input']}"
                elif 'history' in prompt_types and data_args.num_retrieved > 0:  # TODO history only
                    input_text = prompt_generator(x['input'], x['profile'][history_shift:], task)
                else:  # TODO query only
                    input_text = x['input']

                processed_data.append({
                    'p': profile_embs[idmap[x['id']], :],
                    'source': input_text,
                    'target': str(x['output']),
                })
            return processed_data[:sn]


        else:
            # 扩充验证数据

            # 从 evaluate history 采样
            data_path = f'{abs_dir}/rank/{task}/train_questions_rank_{retriever}_merge.json'
            with open(data_path) as f:
                temp_train_data = json.load(f)

            total_sample_num = len(temp_train_data)  # 对齐训练数据量

            data_path = f'{abs_dir}/rank/{task}/dev_questions_rank_{retriever}_merge.json'
            with open(data_path) as f:
                data = json.load(f)

            sample_num = int(total_sample_num / len(data))

            if task in ['IMDB-B', 'YELP-B']:
                uids = set([x['id'] for x in data])
                idmap = {uid: i for i, uid in enumerate(uids)}
            else:
                idmap = {x['id']: i for i, x in enumerate(data)}

            train_data = []
            for x in data:
                cur_profiles = x['profile']

                sample_indices = random.sample(list(range(history_num + 1, len(cur_profiles))),
                                               min(sample_num, len(cur_profiles) - history_num - 1))

                pairs = [apply_task_template(task, cur_profiles[i]) for i in sample_indices]

                profiles = [cur_profiles[i - history_num:i] for i in sample_indices]
                for idx, pair in enumerate(pairs):
                    train_data.append({
                        'p': idmap[x['id']],
                        'source': prompt_generator(pair[0], profiles[idx], task) if data_args.num_retrieved > 0
                        else pair[0],
                        'target': str(pair[1]),
                    })
            return train_data

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train', sn=0)).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    if task in ['IMDB-B', 'YELP-B']:
        my_datasets['dev'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    if tokenize:  # 2 input_ids & label
        my_datasets = my_datasets.map(
            create_preprocessor(tokenizer=tokenizer, max_length=msl),
            batched=True,
            remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_datasets


def load_dataset4da(model_args, data_args, training_args, tokenizer, tokenize=True):
    """
    数据扩充  提高历史数据利用率
    添加 RAG 扰动  & p dropout 处理 鲁棒性
    数据集压缩/过滤  降低训练计算量

    v1 多样性采样(基于答案进行采样)
    """
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    # msl = MSL  # training_args.max_seq_length
    msl = training_args.max_seq_length  # MSL
    history_num = data_args.num_retrieved

    prompt_generator, _ = create_prompt_generator(history_num, retriever,
                                                  is_ranked=True,
                                                  max_length=msl,
                                                  tokenizer=tokenizer)

    def make_data(split, da=False, shift=False):
        assert split in ['train', 'dev', 'test']
        profile_method = 'SBP'  # SBP, SBP_div
        profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
        profile_tensor_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.pt'

        with jsonlines.open(profile_path, 'r') as f:
            profiles = list(f)  # input output
            profiles = [x['output'] for x in profiles]

        device = 'cuda:0'
        profile_embs = None  # profile emb
        if personal_type == 'profile':
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

        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)

        sample_n = 10  # sample + last
        disturb_rate = 0.0  # 20%  加扰动

        if split == 'train':
            data, ids = sample_data(data, profile_embs, task)  # 采样数据
            # TODO 同步修改 profile
        elif split == 'dev':
            ids = [i for i in range(len(data))]
        else:
            ids = [i for i in range(len(data))]

        processed_data = []
        for i, x in enumerate(data):

            x['input'] = trunc_input(tokenizer, x['input'])

            if shift is True:
                history_shift = min(0, len(x['profile']) - history_num)
                histories = x['profile'][history_shift:]
                random.shuffle(histories)
            else:
                history_shift = 0
                histories = x['profile'][history_shift:]

            p = profiles[ids[i]]

            prompt_types = data_args.prompt.split(',')  # default='history,prompt,query'
            if 'profile' in prompt_types and 'history' in prompt_types:  # TODO profile and history
                input_text = prompt_generator(f"**user profile**{p}\n**task**{x['input']}", histories, task)
            elif 'profile' in prompt_types and 'history' not in prompt_types:  # TODO profile only
                input_text = f"**user profile**{p}\n**task**{x['input']}"
            elif 'history' in prompt_types and data_args.num_retrieved > 0:  # TODO history only
                input_text = prompt_generator(x['input'], histories, task)
            else:  # TODO query only
                input_text = x['input']

            # TODO post process
            input_text = input_text.replace('<|begin_of_text|>', ' Task Description: ')
            processed_data.append({
                'p': profile_embs[ids[i], :],
                'source': input_text,
                'target': str(x['output']),
            })
        return processed_data

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    my_datasets['test'] = HFDataset.from_list(make_data('dev', shift=True)).shuffle()

    # if task in ['IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
    #     my_datasets['dev'] = HFDataset.from_list(make_data('test')).shuffle()
    # else:
    #     my_datasets['test'] = my_datasets['dev']

    if tokenize:  # 2 input_ids & label
        my_datasets = my_datasets.map(
            create_preprocessor(tokenizer=tokenizer, max_length=msl),
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
            #
            'LaMP_t_3': "score",
            'LaMP_t_4': "title",
            'LaMP_t_5': "title",
        }
        if his[d[task]] == target_label:
            histories.append(his)
    if len(histories) > 0:
        return histories
    else:
        return x['profile']


def load_dataset4short_cut(model_args, data_args, training_args, tokenizer, tokenize=True):
    """
    数据扩充  提高历史数据利用率
    添加 RAG 扰动  & p dropout 处理 鲁棒性
    数据集压缩/过滤  降低训练计算量

    v1 多样性采样(基于答案进行采样)
    """

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

    def make_data(split, da=False):
        device = 'cuda:0'
        assert split in ['train', 'dev', 'test']
        profile_method = 'SBP'  # SBP, SBP_div
        profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
        profile_tensor_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.pt'

        with jsonlines.open(profile_path, 'r') as f:
            profiles = list(f)  # input output
            profiles = [x['output'] for x in profiles]

        profile_embs = None  # profile emb
        if personal_type == 'profile':
            if os.path.exists(profile_tensor_path):
                profile_embs = torch.load(profile_tensor_path, map_location=torch.device(device))
            else:
                profile_embs = []
                # 加载 嵌入模型
                device = 'cuda:0'
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

        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)

        # TODO load data with reason
        data_path = f'/data/mxl/llm/LLaMa_Factory/data/lamp/{task}_with_reason&profile_train_eval.jsonl'
        with jsonlines.open(data_path, 'r') as f:
            rational_data = list(f)  # instruction  input  output
        if split == 'train':
            outputs = [x['output'] for x in rational_data[:len(data)]]
        else:
            outputs = [str(x['output']) for x in data]
        # outputs = [x['output'] for x in rational_data[-len(data):]
        # TODO insert 2 data

        idmap = {x['id']: i for i, x in enumerate(data)}  # uid 2 -> profile

        disturb_rate = data_args.disturb_rate  # 20%  加扰动
        print(f'Disturb rate: {disturb_rate}')

        processed_data = []
        for i, x in enumerate(data):

            x['input'] = trunc_input(tokenizer, x['input'])

            # TODO 使用相似答案的history
            if split == 'train' and random.random() < disturb_rate:
                xp = filter_sim_label(x, task)
            else:
                xp = x['profile']  # history

            prompt_types = data_args.prompt.split(',')  # default='history,prompt,query'
            if 'profile' in prompt_types and 'history' in prompt_types:  # TODO profile and history
                input_text = prompt_generator(f"**user profile**{profiles[i]}\n**task**{x['input']}",
                                              xp, task)
            elif 'profile' in prompt_types and 'history' not in prompt_types:  # TODO profile only
                input_text = f"**user profile**{profiles[i]}\n**task**{x['input']}"
            elif 'history' in prompt_types and data_args.num_retrieved > 0:  # TODO history only
                input_text = prompt_generator(x['input'], xp, task)
            else:  # TODO query only
                input_text = x['input']

            # TODO post process
            input_text = input_text.replace('<|begin_of_text|>', ' Task Description: ')

            processed_data.append({
                'p': profile_embs[idmap[x['id']], :],
                'source': input_text,
                'target': outputs[i],
            })
        return processed_data

    my_datasets = DatasetDict()
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    # my_datasets['dev_da'] = HFDataset.from_list(make_data('dev', da=True)).shuffle()
    if task in ['IMDB-B', 'YELP-B']:
        my_datasets['dev'] = HFDataset.from_list(make_data('test')).shuffle()
    else:
        my_datasets['test'] = my_datasets['dev']

    if tokenize:  # 2 input_ids & label
        my_datasets = my_datasets.map(
            create_preprocessor(tokenizer=tokenizer, max_length=msl),
            batched=True,
            remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_datasets


def load_dataset4shift_data(model_args, data_args, training_args, tokenizer, tokenize=True):
    personal_type = model_args.personal_type  # uid ,profile

    task = data_args.dataset_name
    retriever = 'contriever'  # data_args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    # msl = MSL  # training_args.max_seq_length
    msl = training_args.max_seq_length  # TODO MSL
    history_num = data_args.num_retrieved

    prompt_generator, _ = create_prompt_generator(history_num, retriever,
                                                  is_ranked=True,
                                                  max_length=msl,
                                                  tokenizer=tokenizer)

    def make_data(split, shift=False):
        assert split in ['train', 'dev', 'test']

        if personal_type == 'profile':
            profiles, profile_embs = load_profile(abs_dir, task, split, retriever)

        data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
        with open(data_path) as f:
            data = json.load(f)  # [:64]

        processed_data = []
        for i, x in tqdm(enumerate(data)):

            if shift is True:
                history_shift = min(0, len(x['profile']) - history_num)
                x['input'] = trunc_input(tokenizer, x['input'])
                histories = x['profile'][history_shift:]
                random.shuffle(histories)
            else:
                history_shift = 0
                x['input'] = trunc_input(tokenizer, x['input'])
                histories = x['profile'][history_shift:]

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

            processed_data.append({
                'p': profile_embs[i, :],
                'source': input_text,
                'target': str(x['output']),
            })

        # raw id ,source target
        # TODO
        #  63% for PPR-B  [0, 80, 320, 1520, 5680]
        #  34% GDRD-B  [238, 626, 1198, 1829, 1033]
        #  33% YELP-B  [2946, 4035, 6692, 14629, 13882]
        #  15% IMDB-B  [485, 350, 337, 201, 708, 793, 1260, 1815, 1009, 1017]
        # 标签分布 [sum([str(i+1)== x['target']  for x in processed_data]) for i in range(5)]
        # 提示答案相似度 sum([x['source'][19]== x['target']  for x in processed_data])/len(processed_data)

        return processed_data

    my_datasets = DatasetDict()  # TODO 注意 和示例答案重合度 85%
    my_datasets['train'] = HFDataset.from_list(make_data('train')).shuffle()
    my_datasets['dev'] = HFDataset.from_list(make_data('dev')).shuffle()
    my_datasets['test'] = HFDataset.from_list(make_data('dev', shift=True)).shuffle()

    # if task in ['IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
    #     my_datasets['test'] = HFDataset.from_list(make_data('test')).shuffle()
    # else:
    #     my_datasets['test'] = my_datasets['dev']

    if tokenize:  # 2 input_ids & label
        my_datasets = my_datasets.map(
            create_preprocessor(tokenizer=tokenizer, max_length=512),
            batched=True,
            remove_columns=['source', 'target'])  # p, input_ids, attention_mask, labels
    return my_datasets
