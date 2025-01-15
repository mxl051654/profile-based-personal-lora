import numpy as np
import evaluate
import nltk
import re

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


# raw for lamp_3
def create_metric_mae_rmse(tokenizer, all_labels):
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

    def compute_metrics(eval_preds):

        """
        transformers 的 compute_metrics 参数是 pred
        EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)

        input = tokenizer([x['input'] for x in train_dataset], padding=True,
                      return_tensors="pt").to(model.device)
        outputs = model.generate(
            # input_ids=input.input_ids,
            # attention_mask=input.attention_mask,
            **input,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        ans = tokenizer.batch_decode(outputs, skip_special_tokens=True)  # TODO 截断输入

        """
        preds, labels = eval_preds
        # if isinstance(preds, tuple):
        #     preds = preds[0]
        preds = np.argmax(preds, axis=-1)

        # input  Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # 解码 labels,去除padding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
        result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}
        """
        raw
        preds, labels = eval_preds
        if isinstance(preds, tuple):
         preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
        result = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}
        return result
        """
        return result

    return compute_metrics


def create_metric_f1_accuracy_chatgpt(all_labels):
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


def create_metric_bleu_rouge_meteor(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # clean_up_tokenization_spaces=True 控制在解码时是否清理空格。例如，使用 BPE tokenizer，解码时可能会出现多余的空格
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        # result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {
            "bleu": result_bleu["score"], "rouge-1": result_rouge["rouge1"], "rouge-2": result_rouge["rouge2"],
            "rouge-L": result_rouge["rougeL"], "rouge-LSum": result_rouge["rougeLsum"],
            # "meteor": result_meteor['meteor']
        }
        return result

    return compute_metrics


def create_metric_bleu_rouge_meteor_chatgpt():
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {
            "bleu": result_bleu["score"], "rouge-1": result_rouge["rouge1"], "rouge-2": result_rouge["rouge2"],
            "rouge-L": result_rouge["rougeL"], "rouge-LSum": result_rouge["rougeLsum"],
            # "meteor": result_meteor['meteor']
        }
        return result

    return compute_metrics


##############
# TODO cus
def cal_metric4lamp_3(result, return_error=False):
    """
    :param result: list of dict{ pred, target}
    """

    def extract_numbers(text):
        # 使用正则表达式查找文本中的数字
        text = text.split('Answer')[-1]
        numbers = re.findall(r'\d+', text)
        # 将匹配的数字转换为整数
        return numbers[0] if len(numbers) > 0 and len(numbers[0]) == 1 else ' '  # [int(num) for num in numbers]

    def create_mapping(x, y):
        try:
            return float(x)  # 可以解析
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):  # 取最远的
                return 1.0
            else:
                return 5.0

    decoded_preds, decoded_labels = [extract_numbers(x['pred']) for x in result], [x['target'] for x in result]

    metric_dir = '/data/mxl/metrics'
    mse_metric = evaluate.load(f'{metric_dir}/mse', module_type='metric', cache_dir=metric_dir)
    mae_metric = evaluate.load(f'{metric_dir}/mae', module_type='metric', cache_dir=metric_dir)

    decoded_preds = [create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)]
    decoded_labels = [create_mapping(x, x) for x in decoded_labels]
    result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared=False)
    metrics = {"mae": result_mae["mae"], "rmse": result_rmse["mse"]}

    if return_error:
        error_index = [i for i, x in enumerate(zip(decoded_preds, decoded_labels)) if x[0] != x[1]]
        return metrics, [x for i, x in enumerate(result) if i in error_index]

    return metrics


def load_metrics(task):
    if task in ['LaMP_3', 'LaMP_t_3', 'IMDB-B', 'YELP-B', 'GDRD-B', 'PPR-B']:
        return cal_metric4lamp_3
    elif task in ['LaMP_1', 'LaMP_2', "LaMP_t_1", 'LaMP_t_2']:
        return create_metric_f1_accuracy
    elif task in ['LaMP_4', 'LaMP_5', 'LaMP_6', 'LaMP_7',
                  'LaMP_t_4', 'LaMP_t_5', 'LaMP_t_6', 'LaMP_t_7']:
        return create_metric_bleu_rouge_meteor
    else:
        print(f'No metric implementation for task {task}')
