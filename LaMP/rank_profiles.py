import torch
from prompts.utils import batchify
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
from prompts.utils import (
    extract_strings_between_quotes,
    extract_after_article,
    extract_after_review,
    extract_after_paper,
    add_string_after_title,
    extract_after_colon,
    extract_after_abstract,
    extract_after_description
)
from rank_bm25 import BM25Okapi
import argparse

parser = argparse.ArgumentParser()
"""
dataset:
https://LaMP-benchmark.github.io/download

Avocado (Personalized Email Subject Generation) dataset
https://catalog.ldc.upenn.edu/LDC2015T03
"""


def load_args():
    """
    --task LaMP_2
    --ranker contriever
    """
    T = 2
    parser.add_argument("--input_data_addr", default=f'../data/LaMP_{T}/train_questions.json')
    parser.add_argument("--output_ranking_addr", default=f'./rank/LaMP_{T}/train_questions_rank.json')
    parser.add_argument("--task", default=f'LaMP_{T}')  # , choices=[f'LaMP_{i}' for i in range(1, 8)])
    parser.add_argument("--ranker", default='bm25', choices=['contriever', 'bm25', 'recency'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_date", action='store_true')
    parser.add_argument("--contriever_checkpoint", default="/data/hf/facebook/contriever")

    args = parser.parse_args()
    return args


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, k, batch_size=16):
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    return [profile[m] for m in topk_indices.tolist()]


def retrieve_top_k_with_bm25(corpus, profile, query, k):
    # 基于 bm25 检索 topk
    tokenized_corpus = [x.split() for x in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    selected_profs = bm25.get_top_n(tokenized_query, profile, n=k)
    return selected_profs


def classification_citation_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x['id'] for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}'
    return corpus, query, ids


def classification_review_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_review(inp)
    return corpus, query, ids


# TODO  for plora IMDB-B YELP-B
def classification_plora_review_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]

    ids = [x['id'] for x in profile]
    query = extract_after_review(inp)
    return corpus, query, ids


def generation_news_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_article(inp)
    return corpus, query, ids


def generation_paper_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids


def parphrase_tweet_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    ids = [x['id'] for x in profile]
    return corpus, query, ids


def generation_avocado_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids


def classification_movies_query_corpus_maker(inp, profile, use_date):
    if use_date:
        # corpus = [f'{x["description"]} date: {x["date"]}' for x in profile]
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        # corpus = [f'{x["description"]}' for x in profile]
        corpus = [f'{x["text"]}' for x in profile]
    # query = extract_after_description(inp)
    query = extract_after_article(inp)  # TODO 似乎数据拼接和 论文 模板不一致
    ids = [x['id'] for x in profile]
    return corpus, query, ids


def main():
    args = load_args()
    task = args.task
    ranker = args.ranker

    with open(args.input_data_addr) as file:
        dataset = json.load(file)  # 单个文件

    rank_dict = dict()

    for data in tqdm.tqdm(dataset):
        inp = data['input']
        profile = data['profile']

        if task in ["LaMP_1", "LaMP_t_1"]:
            corpus, query, ids = classification_citation_query_corpus_maker(inp, profile, args.use_date)
        elif task in ["LaMP_2", "LaMP_t_2"]:
            corpus, query, ids = classification_movies_query_corpus_maker(inp, profile, args.use_date)
        elif task in ["LaMP_3", "LaMP_t_3"]:
            corpus, query, ids = classification_review_query_corpus_maker(inp, profile, args.use_date)

        elif task in ["LaMP_4", "LaMP_t_4"]:
            corpus, query, ids = generation_news_query_corpus_maker(inp, profile, args.use_date)
        elif task in ["LaMP_5", "LaMP_t_5"]:
            corpus, query, ids = generation_paper_query_corpus_maker(inp, profile, args.use_date)
        elif task in ["LaMP_6", "LaMP_t_6"]:
            corpus, query, ids = generation_avocado_query_corpus_maker(inp, profile, args.use_date)
        elif task in ["LaMP_7", "LaMP_t_7"]:
            corpus, query, ids = parphrase_tweet_query_corpus_maker(inp, profile, args.use_date)

        # TODO  类似 LaMP_3
        elif task in ['IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
            corpus, query, ids = classification_plora_review_query_corpus_maker(inp, profile, args.use_date)

        if ranker == "contriever":
            tokenizer = AutoTokenizer.from_pretrained(args.contriever_checkpoint)
            contriver = AutoModel.from_pretrained(args.contriever_checkpoint).to("cuda:0")
            contriver.eval()
            randked_profile = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, len(profile),
                                                            args.batch_size)
        elif ranker == "bm25":
            randked_profile = retrieve_top_k_with_bm25(corpus, profile, query, len(profile))

        elif ranker == "recency":
            profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
            randked_profile = profile[::-1]

        data['profile'] = randked_profile

        rank_dict[data['id']] = [x['id'] for x in randked_profile]

    with open(args.output_ranking_addr, "w") as file:
        json.dump(rank_dict, file)

    # with open(args.output_ranking_addr, "r") as file:
    #     data = json.load(file)


def view_dataset():
    data_dir = '/data/mxl/rlhf/LaMP/data'
    data = dict()
    for i in range(1, 8):
        # task_dir = f"{data_dir}/LaMP_{i}"
        task_dir = f"{data_dir}/LaMP_t_{i}"
        data[i] = dict()
        for fn in ['train_questions', 'train_outputs',
                   'dev_questions', 'dev_outputs',
                   'test_questions', ]:
            try:
                data[i][fn] = json.load(open(f"{task_dir}/{fn}.json", 'rb'))
            except:
                print(f"error {task_dir}/{fn}.json")

    # 对于时序划分，存在用户交际
    # set([x['user_id'] for x in  data[1]['train_questions']]),set([x['user_id'] for x in  data[1]['test_questions']])
    print()


if __name__ == "__main__":
    # view_dataset()
    main()
