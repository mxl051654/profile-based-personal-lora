import argparse
import os
from datetime import datetime


def load_args(make_dir=False):
    """
    --model_name /data/hf/google/flan-t5-base
    --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct

    --method SFT
    --task LaMP_7
    --model_name /data/hf/google/flan-t5-base
    --retriever contriever
    --num_retrieved 4
    --use_profile
    --is_ranked
    """
    parser = argparse.ArgumentParser()

    T = 4

    parser.add_argument("--task", default=f'LaMP_{T}', )  # choices=[f'LaMP_{i}' for i in range(1, 8)])
    parser.add_argument("--split", type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument("--train_data", default=f'./rank/LaMP_{T}/train_questions_rank_bm25_merge.json')
    parser.add_argument("--validation_data", default=f'./rank/LaMP_{T}/dev_questions_rank_bm25__merge.json')
    parser.add_argument("--test_data", default=f'./rank/LaMP_{T}/dev_questions_rank_bm25_merge.json')
    parser.add_argument("--cache_dir", default="./cache")

    # RAG参数
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--use_profile", action="store_true")
    parser.add_argument("--is_ranked", action="store_true")
    parser.add_argument("--num_retrieved", type=int, default=1)

    parser.add_argument("--model_name", default='/data/hf/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--adapter_name_or_path", default=None)
    parser.add_argument("--method", default='SBP', )
    parser.add_argument('--peft_type', default='')
    #                 choices=[
    # 'SFT', 'Lora', 'Direct',
    # 'SBP', 'Iter', 'Hierarchy', 'Recurrent', 'MCTS'])

    # ../ouput/[task]/method [LaMP,ROPG,RSPG]/exp_paras / [logs, cpts, results]
    parser.add_argument("--output_dir", default='./output')  # 存放单次实验 cpt和结果
    parser.add_argument("--profile_dir", type=str, )  # 存放 profile推理结果

    # 训练参数
    parser.add_argument("--max_length", type=int, default=512)  # 太短会导致profile被截断，参考数据统计
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--per_device_batch_size", type=int, default=16)  # 64
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # acc * bs = 64

    args = parser.parse_args()

    abs_dir = '/data/mxl/rlhf/LaMP/LaMP'
    args.train_data = f'{abs_dir}/rank/{args.task}/train_questions_rank_{args.retriever}_merge.json'
    args.validation_data = f'{abs_dir}/rank/{args.task}/dev_questions_rank_{args.retriever}_merge.json'
    if args.task in ['IMDB-B', 'YELP-B']:
        args.test_data = f'{abs_dir}/rank/{args.task}/test_questions_rank_{args.retriever}_merge.json'
        print(f'load data from {args.test_data}')
    else:
        args.test_data = f'{abs_dir}/rank/{args.task}/dev_questions_rank_{args.retriever}_merge.json'

    if args.method in ['SBP', 'SBP_div', 'Iter', 'Hierarchy', 'Recurrent', 'MCTS']:
        """
        sbp series 需要额外保存 profile 数据
        """
        args.profile_dir = f'./profile/{args.task}'
        # 由于结果只依赖 profile,result和profile同名
        args.profile_path = f"{args.split}_{args.method}_{args.model_name.split('/')[-1]}_" \
                            f"{args.retriever}_maxlen_{args.max_length}.jsonl"
        print(f"Profile path : {args.profile_path}")

        if not os.path.exists(args.profile_dir):
            os.makedirs(args.profile_dir)
            print(f'Make dir {args.profile_dir}')
        """
        make sbp result dir 
        """
        args.output_dir = f"../output/{args.task}/{args.method}"
        print(f'Experiment Path :\t {args.output_dir}')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    elif args.method == 'STaR':

        args.output_dir = f"../output/{args.task}/{args.method}"
        print(f'Experiment Path :\t {args.output_dir}')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for dn in ['cpts', 'logs', 'results', 'best_cpt']:
            if not os.path.exists(f"{args.output_dir}/{dn}"):
                os.mkdir(f"{args.output_dir}/{dn}")
                print(f'make dir {args.output_dir}/{dn}')

    else:
        paras = f"{args.retriever}_{args.model_name.split('/')[-1]}_lr_{args.learning_rate}_" + \
                datetime.now().strftime("%Y%m%d-%H%M%S")

        if args.peft_type == '':
            args.output_dir = f"../output/{args.task}/{args.method}/{paras}"
        else:
            args.output_dir = f"../output/{args.task}/{args.method}_{args.peft_type}/{paras}"

        print(f'Experiment Path :\t {args.output_dir}')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for dn in ['cpts', 'logs', 'results', 'best_cpt']:
            if not os.path.exists(f"{args.output_dir}/{dn}"):
                os.mkdir(f"{args.output_dir}/{dn}")
                print(f'make dir {args.output_dir}/{dn}')

    return args
