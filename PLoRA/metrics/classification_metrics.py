import numpy as np
import evaluate
import nltk
import re

metric_dir = '/data/mxl/metrics'
nltk_dir = '/data/mxl/nltk_data'

# print('load nltk (metrics/classification_metrics.py)')
# nltk.download("wordnet", download_dir=nltk_dir)
# nltk.download("punkt", download_dir=nltk_dir)
# nltk.download("omw-1.4", download_dir=nltk_dir)

print('load metrics (metrics/classification_metrics.py)')
f1_metric = evaluate.load(f'{metric_dir}/f1', module_type='metric', cache_dir=metric_dir)
accuracy_metric = evaluate.load(f'{metric_dir}/accuracy', module_type='metric', cache_dir=metric_dir)

mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def create_metric_f1_accuracy(tokenizer, all_labels):
    # f1_metric = evaluate.load(f'{metric_dir}/f1', module_type='metric', cache_dir=metric_dir)
    # accuracy_metric = evaluate.load(f'{metric_dir}/accuracy', module_type='metric', cache_dir=metric_dir)

    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            print(x)
            return -1

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]

        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels,
                                      labels=list(range(len(all_labels))), average="macro")
        result = {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}
        return result

    return compute_metrics


def create_metric_f1_accuracy_bert(all_labels):
    # f1_metric = evaluate.load(f'{metric_dir}/f1', module_type='metric', cache_dir=metric_dir)
    # accuracy_metric = evaluate.load(f'{metric_dir}/accuracy', module_type='metric', cache_dir=metric_dir)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=1)
        result_acc = accuracy_metric.compute(predictions=preds, references=labels)
        result_f1 = f1_metric.compute(predictions=preds, references=labels, labels=list(range(len(all_labels))),
                                      average="macro")
        result = {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}
        return result

    return compute_metrics


def create_metric_mae_rmse_bert(all_labels):
    # mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
    # mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=1)
        result_mae = mae_metric.compute(predictions=preds, references=labels)
        result_rmse = mse_metric.compute(predictions=preds, references=labels, squared=False)
        result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}
        return result

    return compute_metrics


def create_metric_mae_rmse(tokenizer, all_labels):
    # TODO ！！ 1-5/num_class
    num_class = len(all_labels)
    # TODO 对于文本模型  采用 1-5
    all_labels = [float(x) for x in all_labels]

    def extract_numbers(text):
        spt = 'Answer:'
        # 使用正则表达式查找文本中的数字
        # text = text.split(spt)[-1]
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

    def create_mapping(x, y):
        try:
            x = extract_numbers(text=x)
            return float(x)  # 可以解析
        except:
            print('result', x)
            y = float(y)
            if abs(1 - y) > abs(num_class - y):  # 否则取最远的
                return 1.0
            else:
                return num_class

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)  # 去空格

        preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
        labels = [create_mapping(x, x) for x in decoded_labels]
        #
        result_acc = accuracy_metric.compute(predictions=preds, references=labels)
        result_f1 = f1_metric.compute(predictions=preds, references=labels, labels=all_labels, average="macro")
        #
        result_mae = mae_metric.compute(predictions=preds, references=labels)
        result_mse = mse_metric.compute(predictions=preds, references=labels, squared=True)
        result_rmse = mse_metric.compute(predictions=preds, references=labels, squared=False)

        result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"],
                  'acc': result_acc["accuracy"], 'f1': result_f1["f1"], 'mse': result_mse['mse']}
        return result

    return compute_metrics


def create_metric_f1_accuracy_chatgpt(all_labels):
    # f1_metric = evaluate.load("f1")
    # accuracy_metric = evaluate.load("accuracy")

    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            print(x)
            return -1

    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels,
                                      labels=list(range(len(all_labels))), average="macro")
        result = {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}
        return result

    return compute_metrics


def create_metric_mae_rmse_chatgpt(all_labels):
    # mse_metric = evaluate.load("mse")
    # mae_metric = evaluate.load("mae")

    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
        result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}
        return result

    return compute_metrics


def create_cls_metric(all_labels):
    # TODO  对于分类模型  使用 0-4

    def compute_metrics(eval_preds):
        labels = eval_preds.label_ids
        preds = np.argmax(eval_preds.predictions[0], axis=1)

        if isinstance(preds, tuple):
            preds = preds[0]

        result_acc = accuracy_metric.compute(predictions=preds, references=labels)
        result_f1 = f1_metric.compute(predictions=preds, references=labels, labels=list(range(len(all_labels))),
                                      average="macro")
        #
        result_mae = mae_metric.compute(predictions=preds, references=labels)
        result_mse = mse_metric.compute(predictions=preds, references=labels, squared=True)
        result_rmse = mse_metric.compute(predictions=preds, references=labels, squared=False)
        # result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}
        result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"],
                  'acc': result_acc["accuracy"], 'f1': result_f1["f1"], 'mse': result_mse['mse']}
        return result

    return compute_metrics
