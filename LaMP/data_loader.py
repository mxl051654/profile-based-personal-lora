
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator
from metrics.my_metrics import load_metrics


def load_dataset(args):
    """

    dataset {train, dev ,test}
    id source target

    """
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    # collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=args.max_length)

    task = args.task
    if args.use_profile:  # prompt complete with profile
        prompt_generator, contriver = create_prompt_generator(args.num_retrieved, args.retriever, args.is_ranked,
                                                              args.max_length, tokenizer)
    else:
        prompt_generator, contriver = None, None

    metric_func = load_metrics(task)
    greater_is_better = True
    print('process dataset')
    if task == "LaMP_1":
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
    elif task == "LaMP_3":
        train_dataset, labels = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task,
                                                      prompt_generator), get_all_labels(task)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func
        best_metric = "mae"
        greater_is_better = False
    elif task == "LaMP_4":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP_5":
        train_dataset = GeneralSeq2SeqDataset(args.train_data, args.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(args.validation_data, args.use_profile, task, prompt_generator)
        if args.test_data:
            test_dataset = GeneralSeq2SeqDataset(args.test_data, args.use_profile, task, prompt_generator)
        compute_metrics = metric_func(tokenizer=tokenizer)
        best_metric = "rouge-1"
    elif task == "LaMP_7":
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

    dataset = {'train': train_dataset, 'dev': eval_dataset, 'test': test_dataset}
    return dataset, compute_metrics, best_metric
