# nohup python mcts2.py >mcts.out 2>&1 &
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from tqdm import tqdm
from args import load_args
import math
import random

import json
import jsonlines
import random
from collections import defaultdict
import copy
import re
import evaluate
from jinja2 import Template
from my_metrics import load_metrics

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class VllmAGent:
    def __init__(self, model_name_or_path=None, adapter_name_or_path=None, beam_search=False):
        mem = 0.4
        in_max_len = 2048
        out_max_len = 1024
        temp = 0.5

        if adapter_name_or_path:
            self.lora_path = adapter_name_or_path[0]
            self.llm = LLM(
                model=model_name_or_path,
                tokenizer=model_name_or_path,
                gpu_memory_utilization=mem,  # 最大显存预算, vllm kv缓存会占用显存
                max_model_len=in_max_len,
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
                max_model_len=in_max_len,
                disable_log_stats=False,  # 显示进度
                disable_async_output_proc=True,
                max_lora_rank=64,  # default 16
            )
        if not beam_search:
            self.sampling_params = SamplingParams(
                n=1,  # num_return_sequences=repeat_n,
                max_tokens=out_max_len,  # max_new_tokens=128,
                temperature=temp,  # 0 趋向于高频词，容易重复， 1< 原始 softmax, >1 区域均匀，随机
                top_k=20,  # 基于topK单词概率约束候选范围
                length_penalty=1,  # <0 鼓励长句子， >0 鼓励短句子
            )
        else:
            self.sampling_params = SamplingParams(
                use_beam_search=True,
                best_of=3,  # >1
                temperature=temp,
                top_p=1,
                top_k=-1,
                max_tokens=out_max_len,
                length_penalty=0,  # <0 鼓励长句子， >0 鼓励短句子
                n=1,  # num_return_sequences=repeat_n,
            )

    def set_sampling_params(self, **kwargs):
        self.sampling_params = SamplingParams(**kwargs)

    def infer_vllm(self, inputs, instruction=None, chat=False):
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
                                                # 用于回答新对话
                                                add_generation_prompt=True,  # TODO
                                                chat_template=None)
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


metric_dir = '/data/mxl/metrics'
mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

example_template = Template(
    """The input is: 
    {{input}}
    The label is: 
    {{label}}"""
)

error_template = Template(
    """The model's input is: 
    {{input}}
    The model's response is: 
    {{response}}
    The correct label is: 
    {{label}}"""
)

error_feedback_template = Template(
    """
    I'm summarizing user profile for personal generation task.
    My current profile is:
    {{cur_profile}}
    But this profile gets the following errors while infer:
    {{error_string}}
    For each wrong example, carefully examine each question and wrong
    answer step by step, provide comprehensive and different reasons why
    the user profile leads to the wrong answer. 
    At last, based on all these reasons,
    summarize and list all the aspects that improve the profile, keep your answers short and concise.
    """
)

state_transit_template = Template(
    """
    I'm summarizing user profile for personal generation task.
    My current profile is:
    {{cur_profile}}
    But this profile gets the following errors while infer:
    {{error_string}}
    Based on these errors, the problems with with profile and reasons ars:
    {{error_feedback}}
    There is a list of former profiles including the current profile, 
    and each profile is modified from its former profiles:
    {{trajectory_profiles}}
    Based on the above information, please write {{step_num}} new profiles following these guidelines:
    1.The new profiles should solve the current profile's problems.
    2.The new profiles should consider the list of profiles and evolve based on the current profile. 
    3.The new profiles should keep short and concise.
    4.Each new profile should be wrapped with <START> and <END>.
    The new profiles are:
    """
)

state_explore_template = Template(
    """
    I'm summarizing user profile for personal generation/classification task.
    User task histories are: {{history}}
    Based on the above information, please write {{step_num}} new profiles following these guidelines:
    1.The profiles should consider different aspects of depending on the specific task, 
    such as Rating Patterns, Writing Style, Interests Preferences, etc
    2.The profiles be clear, concise and not redundant.
    3.Each proposed profile should be wrapped with <START> and <END>.
    The user profiles are:
    """
)


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


# 定义树节点
class Node:
    def __init__(self, task, state, parent=None):
        self.task = task
        self.state = state  # profile
        self.parent = parent
        self.trajectory_states = []

        self.children = []
        self.visits = 0
        self.score = 0

        if parent is not None:  # 历史（不包括自己）
            self.trajectory_states = parent.trajectory_states + [parent.state]

        self.profile = None

        self.metric_f = load_metrics(task)

        self.simulate_set = []  # 存放 eval history

    def is_fully_expanded(self):
        return len(self.children) > 0 or self.visits > 0

    def has_children(self):
        return len(self.children) > 0

    def best_child(self, total_visit):
        # 如果当前节点有子节点，则选择其中得分最优的子节点
        if self.has_children():
            # c = 0.1
            c = 2.5
            if self.task == 'LaMP_3':  # 越低越好
                best = min(self.children,
                           key=lambda node: node.score - c * pow(total_visit / (node.visits + 1e-6), 0.5))
            else:
                best = max(self.children,
                           key=lambda node: node.score + c * pow(total_visit / (node.visits + 1e-6), 0.5))
        else:
            best = self

        # 如果需要递归查询子节点的最佳选择，可以在此做递归调用
        if best.has_children():  # 假设 has_children() 方法返回是否有子节点
            return best.best_child(total_visit)  # 递归查询子节点中的最佳子节点
        else:
            return best  # 没有子节点时返回当前节点

    # 定义模拟和反思
    def simulate(self, data, llm):
        """
        模拟函数，从用户profile中分析回答是否正确，生成分数。
        """
        self.simulate_set = []  # sample:
        self.simulate_socre = 0

        sample_n = 5  # PromptAgent batch sample_n = 5
        st = random.sample(data['profile'], sample_n)

        qa_pairs = [apply_task_template(self.task, s) for s in st]
        queries = [f"**User Profile**\n{self.state}\n**Task**\n{qa[0]}" for qa in qa_pairs]
        answers = [qa[1] for qa in qa_pairs]
        predicts = llm.infer_vllm(queries, chat=True)

        # scores = eval(predicts, answers)
        scores = [self.eval([p], [a]) for p, a in zip(predicts, answers)]
        print(scores)

        def good_enough(task, scores):
            if task == 'LaMP_3':
                ge = [abs(s) <= 0 for s in scores]
            elif task in ['LaMP_1', 'LaMP_2']:
                ge = []
            else:
                pass

            return ge

        ges = good_enough(self.task, scores)
        self.simulate_set.extend([{
            'profile': self.state,
            'input': qa_pairs[i][0],  # queries[i],
            'output': answers[i],
            'pred': predicts[i],
            'score': scores[i],
            'right': ges[i],
        } for i in range(sample_n)])
        self.score = sum(scores) / len(scores)

    def reflect(self, simulate_sample, llm):

        errors_string = ''
        for i, e in enumerate(simulate_sample):
            error_dict = {'input': e['input'], 'response': e['pred'], 'label': e['output']}
            errors_string += f'Example {i + 1}:' + error_template.render(error_dict)

        error_feedback_in = error_feedback_template.render({
            'cur_profile': self.state,
            'error_string': errors_string,
        })
        error_feedback = llm.infer_vllm([error_feedback_in], chat=True)[0]

        new_profiles_in = state_transit_template.render({
            'cur_profile': self.state,
            'error_string': errors_string,
            'error_feedback': error_feedback,
            'trajectory_states': self.trajectory_states,
        })
        new_profile = llm.infer_vllm([new_profiles_in], chat=True)[0]
        return new_profile

    def explore(self, simulate_sample, llm):

        example_string = ''
        for i, e in enumerate(simulate_sample):
            example_dict = {'input': e['input'], 'label': e['output']}
            example_string += f'Example {i + 1}:' + example_template.render(example_dict)

        explore_input = state_explore_template.render({'history': example_string})

        new_profile = llm.infer_vllm([explore_input], chat=True)[0]
        return new_profile

    def expand(self, data, llm, step):
        self.simulate(data, llm)  # 可以增量添加样本

        error_samples = [sample for sample in self.simulate_set if not sample['right']]

        if len(error_samples) > 1:
            print(f'Reflect')

            th = 0.1 * (2 * step + 1)
            if random.random() < 1:
                new_profile = self.reflect(error_samples, llm)  # 反思
            else:
                new_profile = self.explore(error_samples, llm)  # 随机创新

            new_profiles = [new_p for new_p in new_profile.split('<START>')[1:] if '<END>' in new_p]
            print(f'Expand {len(new_profiles)} new profiles')
            for new_p in new_profiles:
                new_p = new_p.split('<END>')[0]
                child_node = Node(task=self.task, state=new_p, parent=self)
                self.children.append(child_node)

    def backpropagate(self):
        reward = self.score
        node = self
        while node:
            node.score = ((node.visits * node.score) + reward) / (node.visits + 1)
            node.visits += 1
            node = node.parent

    def eval(self, preds, labels):
        score = self.metric_f([{'pred': pred, 'target': label} for pred, label in zip(preds, labels)])

        return score[list(score.keys())[0]]


# 序列化方法
def serialize_tree(node):
    # 将 Node 对象转化为可序列化的字典
    return {
        "task": node.task,
        "state": node.state,
        "trajectory_states": node.trajectory_states,
        "visits": node.visits,
        "score": node.score,
        "children": [serialize_tree(child) for child in node.children]  # 递归序列化子结点
    }


# 反序列化方法
def deserialize_tree(data, parent=None):
    # 从字典重建 Node 对象
    node = Node(
        task=data["task"],
        state=data["state"],
        parent=parent
    )
    node.trajectory_states = data["trajectory_states"]
    node.visits = data["visits"]
    node.score = data["score"]
    for child_data in data["children"]:
        child_node = deserialize_tree(child_data, parent=node)  # 递归反序列化子结点
        node.children.append(child_node)
    return node


# 定义MCTS算法
class MCTS:
    """
    单样本版本，后续考虑batch处理
    """

    def __init__(self, args, data=None, profile=None, root=None):
        self.args = args
        self.task = args.task
        self.data = data
        if root is not None:
            self.root = root
        else:
            self.root = Node(self.task, profile)

        self.metric_f = load_metrics(self.task)

    def select(self):
        return self.root.best_child(self.root.visits)

    def run(self, llm, iterations=5):
        """
        运行MCTS
        """
        for i in range(iterations):
            print(f'iteration {i}')
            # 选择 最佳结点  根据结点的价值（ v + Cx lnN/ni ）  # 结点估值
            node = self.select()

            # 扩展  （采样，选择错误案例进行扩展（反思，生成新状态（profile）））
            node.expand(self.data, llm, i)  # 复用上一步simulate的结果， 添加 node.children

            for child in node.children:
                # 模拟 （采样 数据，测试）
                child.simulate(self.data, llm)  # sample & eval  => eval_set
                # 反向回传 分数
                child.backpropagate()

    def eval(self, llm, state=None):
        if state is None:
            state = self.select().state
        else:
            state = state

        sample_n = 20
        st = random.sample(self.data['profile'], sample_n)

        qa_pairs = [apply_task_template(self.task, s) for s in st]

        queries = [f"**user profile**{state}\n**task**{qa[0]}" for qa in qa_pairs]

        answers = [qa[1] for qa in qa_pairs]

        predicts = llm.infer_vllm(queries, chat=True)

        result = [{'pred': p, 'target': t} for p, t in zip(predicts, answers)]
        score = self.metric_f(result)

        return score[list(score.keys())[0]]

    def cal_gain(self, llm, state=None):

        new_state = self.select().state

        sample_n = 20
        st = random.sample(self.data['profile'], sample_n)
        qa_pairs = [apply_task_template(self.task, s) for s in st]
        queries = [f"**user profile**{state}\n**task**{qa[0]}" for qa in qa_pairs]
        answers = [qa[1] for qa in qa_pairs]

        predicts = llm.infer_vllm(queries, chat=True)
        result = [{'pred': p, 'target': t} for p, t in zip(predicts, answers)]
        scores = self.metric_f(result)
        score = scores[list(scores.keys())[0]]

        qa_pairs = [apply_task_template(self.task, s) for s in st]
        queries = [f"**user profile**{new_state}\n**task**{qa[0]}" for qa in qa_pairs]
        answers = [qa[1] for qa in qa_pairs]

        predicts = llm.infer_vllm(queries, chat=True)
        result = [{'pred': p, 'target': t} for p, t in zip(predicts, answers)]
        scores = self.metric_f(result)
        new_score = scores[list(scores.keys())[0]]

        return {'raw_score': score, 'new_score': new_score}

    def export(self):
        serialized_data = serialize_tree(self.root)
        return serialized_data


def load_raw_data(args, split='dev'):
    task = args.task  # args.dataset_name
    retriever = args.retriever
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    data_path = f'{abs_dir}/rank/{task}/{split}_questions_rank_{retriever}_merge.json'
    with open(data_path) as file:
        data = json.load(file)
    return data


def load_profiles(args, split='dev', profile_method='SBP'):
    assert profile_method in ['SBP', 'SBP_div', 'MCTS', 'Hierarchy', 'Iter']
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    task = args.task
    retriever = 'contriever'
    # profile_method = 'SBP'  # SBP, SBP_div Iter
    profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'
    with jsonlines.open(profile_path, 'r') as f:
        profiles = list(f)  # input output
        if profile_method == 'MCTS':
            ret_profiles = []
            for x in profiles:
                if x['raw_score'] < x['score']:  # for LaMP_3 mae
                    ret_profiles.append(x['raw_profile'])
                else:
                    ret_profiles.append(x['output'])
        else:
            ret_profiles = [x['output'] for x in profiles]
    return ret_profiles


def load_trees(args, split='dev', profile_method='SBP'):
    assert profile_method in ['SBP', 'SBP_div', 'MCTS', 'Hierarchy', 'Iter']
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    task = args.task
    retriever = 'contriever'
    # profile_method = 'SBP'  # SBP, SBP_div Iter
    profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024_tree.jsonl'
    with jsonlines.open(profile_path, 'r') as f:
        profiles = list(f)  # input output
    return profiles


def save_trees(data, args, split='dev', profile_method='MCTS'):
    assert profile_method in ['SBP', 'SBP_div', 'MCTS', 'Hierarchy', 'Iter']
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    task = args.task
    retriever = 'contriever'
    profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024_tree.jsonl'

    with jsonlines.open(profile_path, 'w') as f:
        f.write_all(data)
        print(f'save profiles at {profile_path}')


def save_profiles(data, args, split='dev', profile_method='MCTS'):
    assert profile_method in ['SBP', 'SBP_div', 'MCTS', 'Hierarchy', 'Iter']
    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    task = args.task
    retriever = 'contriever'
    # profile_method = 'SBP'  # SBP, SBP_div Iter
    profile_path = f'{abs_dir}/profile/{task}/{split}_{profile_method}_Meta-Llama-3-8B-Instruct_{retriever}_maxlen_1024.jsonl'

    with jsonlines.open(profile_path, 'w') as f:
        f.write_all(data)
        print(f'save profiles at {profile_path}')


def MCTS_main():
    """
    /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
     /data/hf/meta-llama/Meta-Llama-3-1B-Instruct


        nohup python sbp.py
        --task LaMP_3
        --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct
        --retriever contriever
        --use_profile
        --is_ranked
        --max_length 2048
        --method MCTS
        """

    # 初始化数据和MCTS
    # data = [
    #     {"id": 1, "input": "Q1", "output": "A1", "profile": [{"id": 1, "text": "likes A", "score": 30}]},
    #     {"id": 2, "input": "Q2", "output": "A2", "profile": [{"id": 2, "text": "likes B", "score": 40}]},
    #     # ...更多数据
    # ]
    # root_state = {"input": "Start", "output": None, "profile": [{"id": 0, "text": "neutral", "score": 50}]}

    args = load_args()
    split = 'dev'
    dataset = load_raw_data(args, split)
    profiles = load_profiles(args, split, profile_method='SBP_div')

    llm = VllmAGent(args.model_name)

    ss_scores = []
    es_scores = []

    tree_jsons = []
    new_profiles = []
    for data, profile in zip(dataset[:], profiles[:]):
        """
        data  id, input, output ,profile(id, text, score)
        profile  'xx'
        """
        mcts = MCTS(args, data, profile)

        raw_score = mcts.eval(llm)
        ss_scores.append(raw_score)

        mcts.run(llm)

        new_score = mcts.eval(llm)
        es_scores.append(new_score)

        print(f'### mcts gain {es_scores[-1] - ss_scores[-1]}')

        tree_jsons.append(mcts.export())
        new_profiles.append({
            'id': data['id'],
            'output': mcts.select().state, 'score': new_score,
            'raw_profile': profile, 'raw_score': raw_score})

    mean_ss = sum(ss_scores) / len(ss_scores)
    mean_es = sum(es_scores) / len(es_scores)
    print(f"### mean ss_score {mean_ss}  \n ### mean es_score {mean_es}")
    print(f"### final gain {mean_es - mean_ss}")

    save_profiles(new_profiles, args, split, profile_method='MCTS')
    save_trees(tree_jsons, args, split, profile_method='MCTS')


def compare():
    args = load_args()
    split = 'dev'
    dataset = load_raw_data(args, split)

    llm = VllmAGent(args.model_name)
    # llm = None

    # for profile_method in ['None', 'SBP', 'SBP_div', 'MCTS', 'Hierarchy', 'Iter']:
    for profile_method in ['None', 'SBP', 'SBP_div', 'Hierarchy', 'Iter']:

        if profile_method != 'None':
            profiles = load_profiles(args, split, profile_method=profile_method)
        else:
            profiles = [''] * len(dataset)

        metric_f = load_metrics(args.task)

        # TODO for expand eval
        # eval_num = len(dataset)
        # queries = []
        # answers = []
        # for data, profile in zip(dataset[:eval_num], profiles[:eval_num]):
        #     """
        #     data  id, input, output ,profile(id, text, score)
        #     profile  'xx'
        #     """
        #     sample_n = 20
        #     st = random.sample(data['profile'], sample_n)
        #     qa_pairs = [apply_task_template(args.task, s) for s in st]
        #     queries = queries + [f"**user profile**{profile}\n**task**{qa[0]}" for qa in qa_pairs]
        #
        #     answers = answers + [qa[1] for qa in qa_pairs]

        # for raw eval set
        queries = [f"**user profile**{p}\n**task**{x['input']}" for p, x in zip(profiles, dataset)]
        answers = [x['output'] for x in dataset]

        predicts = llm.infer_vllm(queries, chat=True)
        result = [{'pred': p, 'target': t} for p, t in zip(predicts, answers)]

        score = metric_f(result)
        print(f'{profile_method} score : {score}')


def analysis():
    args = load_args()
    split = 'dev'
    trees = load_trees(args, split='dev', profile_method='MCTS')

    dataset = load_raw_data(args, split)
    profiles = load_profiles(args, split, profile_method='SBP_div')

    # trees = [MCTS(args, root=deserialize_tree(t)) for t in trees]

    llm = VllmAGent(args.model_name)

    ss_scores = []
    es_scores = []

    new_profiles = []
    for i in range(len(trees)):
        """
        data  id, input, output ,profile(id, text, score)
        profile  'xx'
        """
        data = dataset[i]
        profile = profiles[i]
        mcts = MCTS(args, data=data, root=deserialize_tree(trees[i]))

        s = mcts.cal_gain(llm, profile)
        ss_scores.append(s['raw_score'])
        es_scores.append(s['new_score'])

        print(f'### mcts gain {es_scores[-1] - ss_scores[-1]}')

        new_profiles.append({
            'id': data['id'],
            'output': mcts.select().state, 'score': s['new_score'],
            'raw_profile': profile, 'raw_score': s['raw_score']})

    mean_ss = sum(ss_scores) / len(ss_scores)
    mean_es = sum(es_scores) / len(es_scores)
    print(f"### mean ss_score {mean_ss}  \n ### mean es_score {mean_es}")
    print(f"### final gain {mean_es - mean_ss}")

    save_profiles(new_profiles, args, split, profile_method='MCTS')


def after_analysis():
    args = load_args()
    split = 'dev'
    dataset = load_raw_data(args, split)

    profiles = load_profiles(args, split, profile_method='MCTS')

    metric_f = load_metrics(args.task)

    llm = VllmAGent(args.model_name)

    queries = [f"**user profile**{p}\n**task**{x['input']}" for p, x in zip(profiles, dataset)]
    answers = [x['output'] for x in dataset]

    predicts = llm.infer_vllm(queries, chat=True)
    result = [{'pred': p, 'target': t} for p, t in zip(predicts, answers)]

    score = metric_f(result)
    print(f'score : {score}')


if __name__ == '__main__':
    # MCTS_main()

    # compare()

    # analysis()

    after_analysis()
