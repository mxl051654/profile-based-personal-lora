import os

os.environ['CUDA_HOME'] = '/usr/local/cuda-11.3'
import nltk
import shutil

nltk.data.path.append('/data/mxl/nltk_data')
# pip install --upgrade datasets multiprocess
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback

from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor

from data.datasets import (
    get_all_labels,
    GeneralSeq2SeqDataset,
    create_preprocessor,
    convert_to_hf_dataset
)
from prompts.prompts import create_prompt_generator
from datetime import datetime
from peft import (
    get_peft_model, MultitaskPromptTuningConfig, TaskType, MultitaskPromptTuningInit,
    LoraConfig,
)


def module_test():
    """ test load local data """
    from datasets import load_dataset
    # print(os.path.exists('../../../huggingface/cnn_dailymail/3.0.0'))
    # dataset = load_dataset('../../../huggingface/cnn_dailymail/3.0.0')
    ## dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir='../../../huggingface/cnn_dailymail/3.0.0/')

    """ test laod local model """
    # from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # print(os.getcwd())
    # model_name =  '../../../huggingface/google-t5/t5-base'
    # policy_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    nltk_dir = '/data/mxl/nltk_data'
    metric_dir = '/data/mxl/metrics'

    """ test load nltk_data from local """

    # 手动下载
    # print('down load nltk')
    # nltk.download("wordnet", download_dir=nltk_dir)
    # nltk.download("punkt", download_dir=nltk_dir)
    # nltk.download("omw-1.4", download_dir=nltk_dir)
    nltk.data.path.append(nltk_dir)

    import evaluate
    # view all evaluate module
    # res = evaluate.list_evaluation_modules(  # evaluate 几乎所有操作都需要从huggingface加载
    #     module_type="metric",
    #     include_community=False,
    #     with_details=True)
    # print(res)

    m = evaluate.load(f'{metric_dir}/meteor', module_type='metric', cache_dir=metric_dir)

    m = evaluate.load(f'{metric_dir}/rouge', module_type='metric', cache_dir=metric_dir)  # pip install -U rouge-score
    # print(m.inputs_description)  # 查看输入格式
    res = m.compute(predictions=["hello there", 'hh'], references=["hello there ha", 'hh'])
    print(res)

    # m = evaluate.load('./metrics/bertscore', module_type='metric', cache_dir='./metrics')
    # res = m.compute(predictions=["hello chicago"], references=["hello chicago hei"], lang='en')
    # print(res)  # 需要加载模型

    m = evaluate.load(f'{metric_dir}/bleu', module_type='metric', cache_dir=metric_dir)
    res = m.compute(predictions=["hello chicago"], references=["hello chicago hei"])
    print(res)

    m = evaluate.load(f'{metric_dir}/sacrebleu', module_type='metric', cache_dir=metric_dir)

    """  多维指标的雷达图  """
    # import evaluate
    # from evaluate.visualization import radar_plot
    # import matplotlib.pyplot as plt
    #
    # data = [
    #     {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
    #     {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
    #     {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
    #     {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
    # ]
    #
    # model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
    # plot = radar_plot(data=data, model_names=model_names)
    # plot.show()
    # plt.show()


def main():
    from args import load_args
    args = load_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    if args.peft_type == 'LORA':
        peft_config = LoraConfig(
            lora_alpha=16,  # 常数，类似于学习率
            lora_dropout=0.1,
            target_modules=['q', 'v'],
            r=32,
            bias="none",
            task_type='SEQ_2_SEQ_LM',  # "CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=args.max_length)

    task = args.task
    if args.use_profile:
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
    elif task == "LaMP_6":
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

    # 使用数据缓存需要注意 transformers版本
    train_dataset = convert_to_hf_dataset(train_dataset, cache_dir=args.cache_dir).map(
        create_preprocessor(tokenizer=tokenizer, max_length=args.max_length), batched=True)

    eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir=args.cache_dir).map(
        create_preprocessor(tokenizer=tokenizer, max_length=args.max_length), batched=True)

    if args.test_data:
        test_dataset = convert_to_hf_dataset(test_dataset, cache_dir=args.cache_dir).map(
            create_preprocessor(tokenizer=tokenizer, max_length=args.max_length), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{args.output_dir}/cpts',
        logging_dir=f'{args.output_dir}/logs',
        do_train=True,
        do_eval=True,
        # evaluation_strategy="epoch",  # 评估
        eval_strategy="epoch",  # 评估
        save_strategy="epoch",  # 保存检查点
        logging_steps=50,  # 日志步长
        eval_accumulation_steps=1,
        load_best_model_at_end=True,  # 训练结束加载最佳模型
        metric_for_best_model=best_metric,  # 评估指标
        greater_is_better=greater_is_better,  # 指标越大越好

        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,

        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        generation_num_beams=args.generation_num_beams,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print('start train')
    trainer.train()

    if args.test_data:
        results = trainer.evaluate(test_dataset)  # TODO:修改，在不训练时加载CPT
        print(results)

        with open(f'{args.output_dir}/results/dev_output.json', 'w') as file:
            json.dump(results, file, indent=4)

    # 保存最佳模型
    trainer.save_model(f'{args.output_dir}/best_cpt', )
    # 删除
    # os.removedirs(f'{args.output_dir}/cpts')
    shutil.rmtree(f'{args.output_dir}/cpts')  # 递归删除


if __name__ == "__main__":
    # module_test()
    main()
