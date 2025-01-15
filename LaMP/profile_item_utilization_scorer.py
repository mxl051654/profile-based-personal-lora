import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import nltk

nltk.data.path.append('/data/mxl/nltk_data')

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    AutoModelForCausalLM
# from transformers.models.llama import LlamaTokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from data.datasets import get_all_labels, GeneralSeq2SeqForScoreGenerationDataset, create_preprocessor_scores, \
    convert_to_hf_dataset
from prompts.prompts import create_prompt_generator
import tqdm
import datasets
import json


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_name", required=True,
                        help='address to the model that should be used for feedback generation')
    parser.add_argument("--task", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--profile_size", type=int, default=10, help='profile sample size for a query')

    return parser.parse_args()


def main():
    opts = load_args()
    print('load_model')
    model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=opts.max_length)

    print('load dataset')
    task = opts.task
    prompt_generator, contriver = create_prompt_generator(1, "bm25", True, opts.max_length, tokenizer)

    if task in ["LaMP_1", "LaMP_t_1"]:
        labels = get_all_labels(task)
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
    elif task in ["LaMP_2", "LaMP_t_2"]:
        labels = get_all_labels(task)
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels)
    elif task in ["LaMP_3", "LaMP_t_3"]:
        labels = get_all_labels(task)
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels)
    elif task in ["LaMP_4", "LaMP_t_4"]:
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
    elif task in ["LaMP_5", "LaMP_t_5"]:
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    elif task in ["LaMP_6", "LaMP_t_6"]:
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    elif task in ["LaMP_7", "LaMP_t_7"]:
        eval_dataset = GeneralSeq2SeqForScoreGenerationDataset(opts.train_data, True, task, prompt_generator,
                                                               opts.profile_size)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    eval_dataset = convert_to_hf_dataset(eval_dataset, opts.cache_dir).map(
        create_preprocessor_scores(tokenizer=tokenizer, max_length=opts.max_length),
        batched=True, )

    # if contriver:
    #     contriver = contriver.to("cpu")

    training_args = Seq2SeqTrainingArguments(
        output_dir=opts.output_dir,
        do_eval=True,
        per_device_eval_batch_size=1,
        generation_num_beams=opts.generation_num_beams,
        predict_with_generate=True,
        eval_accumulation_steps=1,
        generation_max_length=opts.generation_max_length
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    results_dict = dict()

    for i, x in enumerate(tqdm.tqdm(eval_dataset)):
        if i < opts.start_index:
            continue
        if i >= opts.end_index and opts.end_index != -1:
            break

        metrics = trainer.predict(datasets.Dataset.from_list([x])).metrics

        results_dict[f"{x['id_1']}-{x['id_2']}"] = {k.replace("test_", ''): v for k, v in metrics.items()}

    with open(os.path.join(opts.output_dir, f"scores_{opts.start_index}_{opts.end_index}.json"), "w") as file:
        json.dump(results_dict, file, indent=4)


if __name__ == "__main__":
    """
    Feedback Generation using LLM for Items in the User Profile
    The first step is to collect feedback from the LLM using the following script:

    cd LaMP
    python profile_item_utilization_scorer.py \
        --train_data /*address to sorted training data using the previous steps*/ \
        --model_name /*address to the model that should be used for feedback generation*/ \
        --task /*name of the task [LaMP_1, LaMP_2, ..., LaMP_7]*/ \
        --output_dir /*output directory to save results */ \
        --profile_size /*number of top k items from user profile to get feedback for them*/
        
    
    --task LaMP_1
    --train_data ./rank/LaMP_1/train_questions_rank_bm25_merge.json
    --output_dir ./feedback/LaMP_1/train_bm25_flan-t5-base
    --model_name "/data/hf/google/flan-t5-base"
    --profile_size 1
        
    """
    main()
