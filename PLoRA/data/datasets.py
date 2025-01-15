import os

import random
import json
import datasets
from collections import defaultdict
from torch.utils.data import Dataset


def get_all_labels(task):
    if task in ["LaMP_1", 'LaMP_t_1']:
        return ["[1]", "[2]"]
    elif task == "LaMP_2":
        return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic',
                'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
    elif task in ["LaMP_3", 'LaMP_t_3']:
        return ["1", "2", "3", "4", "5"]
    # TODO
    elif task == 'IMDB-B':
        return [str(i + 1) for i in range(10)]
    elif task in ['YELP-B', 'GDRD-B', 'PPR-B']:
        return [str(i + 1) for i in range(5)]
    else:
        return []


def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        # model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True,)
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True, padding=True)

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

        # model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True,)
        model_inputs = tokenizer(inputs,  # 输入
                                 text_target=targets,  # 输出
                                 max_length=max_length, truncation=True, padding=True)

        bge_query = bge_tokenizer(inputs, return_tensors='pt',
                                  padding='max_length', truncation=True,
                                  max_length=512)
        #
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


def convert_to_hf_dataset(dataset, cache_dir):
    def gen():
        for idx in range(len(dataset)):
            yield dataset[idx]

    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)


class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt=None) -> None:
        super().__init__()
        # 加载数据
        with open(data_addr) as file:
            self.data = json.load(file)  # [:10]
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), \
            "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt

    def __getitem__(self, index):
        if self.use_profile:
            return {
                "id": self.data[index]['id'],
                "source": self.create_prompt(self.data[index]['input'], self.data[index]['profile'], self.task),
                "target": self.data[index]['output']
            }
        else:
            return {
                "id": self.data[index]['id'],
                "source": self.data[index]['input'],
                "target": self.data[index]['output']
            }

    def __len__(self):
        return len(self.data)


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


class GeneralSeq2SeqDatasetWH(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt=None) -> None:
        super().__init__()
        # 加载数据
        with open(data_addr) as file:
            self.data = json.load(file)  # [:10]
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), \
            "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt

        self.profiles = self.sample_profiles()

    def sample_profiles(self, n=10):
        sampled_profiles = []
        for item in self.data:
            # profiles = item['profile']  # id, text, score
            sp = sample_profiles_by_score(item['profile'], n)
            sp = [f"Score {x['score']} for {x['text']}" for x in sp]
            sampled_profiles.append(sp)
        return sampled_profiles

    def __getitem__(self, index):
        if self.use_profile:
            return {
                "id": self.data[index]['id'],
                "source": self.create_prompt(self.data[index]['input'], self.data[index]['profile'], self.task),
                "target": self.data[index]['output'],
                "profile": self.profiles[index],
            }
        else:
            return {
                "id": self.data[index]['id'],
                "source": self.data[index]['input'],
                "target": self.profiles[index],
            }

    def __len__(self):
        return len(self.data)


class GeneralSeq2SeqForScoreGenerationDataset(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt=None, max_prof_size=-1) -> None:
        super().__init__()
        with open(data_addr) as file:
            self.data = json.load(file)
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), \
            "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt
        self.max_prof_size = max_prof_size
        self.size = 0
        self.index_dict = dict()
        for i, x in enumerate(self.data):
            for j, y in enumerate(x['profile']):
                if max_prof_size == -1 or j < self.max_prof_size:
                    self.index_dict[self.size] = (i, j)
                    self.size += 1

    def __getitem__(self, index):

        self.use_profile = True
        i, j = self.index_dict[index]
        if self.use_profile:
            return {
                "source": self.create_prompt(self.data[i]['input'], [self.data[i]['profile'][j]], self.task),
                "target": self.data[i]['output'],
                "id_1": self.data[i]['id'],
                "id_2": self.data[i]['profile'][j]['id']
            }
        else:
            return {
                "source": self.data[index]['input'],
                "target": self.data[index]['output']
            }

    def __len__(self):
        return self.size
