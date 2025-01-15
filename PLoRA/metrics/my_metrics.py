import numpy as np
import evaluate
import nltk
import re


def get_all_labels(task):
    if task == "LaMP_1":
        return ["[1]", "[2]"]
    elif task == "LaMP_2":
        return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic',
                'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
    elif task == "LaMP_3":
        return ["1", "2", "3", "4", "5"]
    elif task == "LaMP_4":
        return []
    elif task == "LaMP_5":
        return []
    elif task == "LaMP_6":
        return []
    elif task == "LaMP_7":
        return []
    # TODO
    elif task == 'IMDB-B':
        return [str(i + 1) for i in range(10)]
    elif task == 'YELP-B':
        return [str(i + 1) for i in range(5)]


metric_dir = '/data/mxl/metrics'
nltk_dir = '/data/mxl/nltk_data'

# import nltk
# nltk.data.path.append('/data/mxl/nltk_data')
# # pip install --upgrade datasets multiprocess
# print('load nltk (metrics/classification_metrics.py)')
# nltk.download("wordnet", download_dir=nltk_dir)
# nltk.download("punkt", download_dir=nltk_dir)
# nltk.download("omw-1.4", download_dir=nltk_dir)

print('load metrics (metrics/classification_metrics.py)')
f1_metric = evaluate.load(f'{metric_dir}/f1', module_type='metric', cache_dir=metric_dir)
accuracy_metric = evaluate.load(f'{metric_dir}/accuracy', module_type='metric', cache_dir=metric_dir)
mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)
print('load metrics (metrics/generation_metrics.py)')
bleu_metric = evaluate.load(f'{metric_dir}/sacrebleu', module_type='metric', cache_dir=metric_dir)
rouge_metric = evaluate.load(f'{metric_dir}/rouge', module_type='metric', cache_dir=metric_dir)

# meteor 依赖 nltk.download("wordnet", download_dir='./nltk_data')
# meteor_metric = evaluate.load(f'{metric_dir}/meteor/meteor.py', module_type='metric', cache_dir=metric_dir)

spt = '**Answer**'


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def crate_metric4lamp12(all_labels):
    def cal_metric4lamp_12(results):

        def extract_numbers(text):
            # 使用正则表达式查找文本中的数字
            text = text.split(spt)[-1]
            numbers = re.findall(r'\d+', text)
            # 将匹配的数字转换为整数
            number = numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '
            return f'[{number}]'

        def create_mapping(x):
            try:
                return all_labels.index(x)
            except:
                print(x)
                return -1

        preds, labels = [x['pred'] for x in results], [x['target'] for x in results]

        decoded_preds, decoded_labels = postprocess_text(preds, labels)

        decoded_preds = [create_mapping(extract_numbers(x)) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]

        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels,
                                      labels=list(range(len(all_labels))), average="macro")
        metric = {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}
        return metric

    return cal_metric4lamp_12


def cal_metric4lamp_3(result, return_error=False):
    """
    :param result: list of dict{ pred, target}
    """

    def extract_numbers(text):
        # 使用正则表达式查找文本中的数字
        text = text.split(spt)[-1]
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

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

    preds, labels = [extract_numbers(x['pred']) for x in result], [x['target'] for x in result]

    metric_dir = '/data/mxl/metrics'
    mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
    mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

    decoded_preds = [create_mapping(x, y) for x, y in zip(preds, labels)]
    decoded_labels = [create_mapping(x, x) for x in labels]
    result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
    metrics = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}

    if return_error:
        error_index = [i for i, x in enumerate(zip(decoded_preds, decoded_labels)) if x[0] != x[1]]
        return metrics, [x for i, x in enumerate(result) if i in error_index]

    return metrics


def cal_metric4lamp_4567(results):
    def extract_answer(text):
        # 使用正则表达式查找文本中的数字
        text = text.split(spt)[-1]
        return text

    preds, labels = [x['pred'] for x in results], [x['target'] for x in results]
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    result = {
        "bleu": result_bleu["score"], "rouge-1": result_rouge["rouge1"], "rouge-2": result_rouge["rouge2"],
        "rouge-L": result_rouge["rougeL"], "rouge-LSum": result_rouge["rougeLsum"],
        # "meteor": result_meteor['meteor']
    }
    return result


def load_metrics(task):
    if task == 'LaMP_3':
        return cal_metric4lamp_3
    elif task in ['LaMP_1', 'LaMP_2']:
        return crate_metric4lamp12(get_all_labels(task))
    elif task in ['LaMP_4', 'LaMP_5', 'LaMP_6', 'LaMP_7']:
        return cal_metric4lamp_4567
    else:
        print(f'No metric implementation for task {task}')
