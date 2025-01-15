import numpy as np
import evaluate

metric_dir = '/data/mxl/metrics'
print('load metrics (metrics/generation_metrics.py)')
bleu_metric = evaluate.load(f'{metric_dir}/sacrebleu', module_type='metric', cache_dir=metric_dir)
rouge_metric = evaluate.load(f'{metric_dir}/rouge', module_type='metric', cache_dir=metric_dir)


# meteor 依赖 nltk.download("wordnet", download_dir='./nltk_data')
# meteor_metric = evaluate.load(f'{metric_dir}/meteor/meteor.py', module_type='metric', cache_dir=metric_dir)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def create_metric_bleu_rouge_meteor(tokenizer):
    # bleu_metric = evaluate.load("sacrebleu")
    # rouge_metric = evaluate.load('rouge')
    # meteor_metric = evaluate.load('meteor')

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

        result = {"bleu": result_bleu["score"], "rouge-1": result_rouge["rouge1"], "rouge-2": result_rouge["rouge2"],
                  "rouge-L": result_rouge["rougeL"], "rouge-LSum": result_rouge["rougeLsum"],
                  # "meteor": result_meteor['meteor']
                  }
        return result

    return compute_metrics


def create_metric_bleu_rouge_meteor_chatgpt():
    # bleu_metric = evaluate.load("sacrebleu")
    # rouge_metric = evaluate.load('rouge')
    # meteor_metric = evaluate.load('meteor')

    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {"bleu": result_bleu["score"], "rouge-1": result_rouge["rouge1"], "rouge-2": result_rouge["rouge2"],
                  "rouge-L": result_rouge["rougeL"], "rouge-LSum": result_rouge["rougeLsum"],
                  # "meteor": result_meteor['meteor']
                  }
        return result

    return compute_metrics
