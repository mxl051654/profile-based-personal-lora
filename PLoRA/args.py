import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import HfArgumentParser, set_seed
from trl import SFTConfig, SFTTrainer

# TODO https://github.com/huggingface/peft/blob/main/examples/sft/train.py


model2path = {
    'bert': '',
    'roberta': '',
    'flan_t5': '/data/hf/google/flan-t5-base',
    # 'llama': '/data/hf/meta-llama/Meta-Llama-3-8B-Instruct',
    'llama': '/data/hf/meta-llama/Llama-3.2-1B-Instruct',
}


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    # p - peft
    user_size: Optional[int] = field(default=0)  # 用户 Embedding num
    user_feature_dim: Optional[int] = field(default=768)  # 用户特征维度
    personal_type: Optional[str] = field(default='')  # uid, profile
    expert_num: Optional[int] = field(default=1)  # >1 for moe_lora
    module_mask_percent: Optional[int] = field(default=0)  # 0-100

    inject_method: Optional[str] = field(default='add')  # add cat

    # cls
    num_labels: Optional[int] = field(default=5)

    # lora
    adapter_name_or_path: Optional[str] = field(default=None)
    train_method: Optional[str] = field(default='sft')
    peft_type: Optional[str] = field(default='lora')  # lora, p-tuningv2, plora
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    # all or like  e1,e2,e3 or d1 d2
    lora_target_layers: Optional[str] = field(default='all')

    # quant
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="LaMP_3",
        metadata={"help": "The preference dataset to use."},
    )
    #  ['SBP', 'SBP_div', 'Iter', 'Hierarchy', 'Recurrent', 'MCTS']
    profile_method: Optional[str] = field(default='SBP')
    profile_dir: Optional[str] = field(default="./profiles")

    prompt: Optional[str] = field(default='history,prompt,query', )
    retriever: Optional[str] = field(default='contriever')  # bm25, contriever
    num_retrieved: Optional[int] = field(default=1)  # 1-4
    suffix: Optional[str] = field(default='')

    disturb_rate: Optional[float] = field(default=0)

    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def process_args(make_dir=False):
    """
    SFTConfig(TrainingArguments)
    dataset_text_field: Optional[str] = None
    max_seq_length: Optional[int] = None
    dataset_batch_size: int = 1000
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO 后处理 training args
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    training_args.dataset_kwargs = {
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    }
    set_seed(training_args.seed)

    # TODO
    if model_args.peft_type != 'MOE':
        model_args.expert_num = 1

    # TODO
    if make_dir:
        make_exp_dirs(model_args, data_args, training_args)

    return model_args, data_args, training_args


def make_exp_dirs(model_args, data_args, training_args):
    paras = f"{data_args.retriever}_{model_args.model_name_or_path.split('/')[-1]}_" \
            f"lr_{training_args.learning_rate}_{model_args.personal_type}_" + \
            data_args.suffix + \
            datetime.now().strftime("%Y%m%d-%H%M%S")

    training_args.output_dir = f"../output/{data_args.dataset_name}/" \
                               f"{model_args.train_method}_{model_args.peft_type}/{paras}"

    print(f'Experiment Path :\t {training_args.output_dir}')
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    for dn in ['cpts', 'logs', 'results', 'best_cpt']:
        if not os.path.exists(f"{training_args.output_dir}/{dn}"):
            os.mkdir(f"{training_args.output_dir}/{dn}")
            print(f'make dir {training_args.output_dir}/{dn}')
