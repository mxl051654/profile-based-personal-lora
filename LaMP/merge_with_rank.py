import json
import argparse


def merge(inps, outs, ranks):
    for inp in inps:
        # for o in outs:
        for o in outs['golds']:
            if o['id'] == inp['id']:
                output = o['output']
                break
        new_profile = []
        for x in ranks[inp['id']]:
            for y in inp['profile']:
                if y['id'] == x:
                    new_profile.append(y)
                    break
        inp['profile'] = new_profile
        inp['output'] = output
    return inps


def load_args():
    parser = argparse.ArgumentParser()
    T = 7
    parser.add_argument("--lamp_questions_addr", default=f'../data/LaMP_{T}/train_questions.json', )
    parser.add_argument("--lamp_output_addr", default=f'../data/LaMP_{T}/train_outputs.json', )
    parser.add_argument("--profile_ranking_addr", default=f"./rank/LaMP_{T}/train_questions_rank.json")
    parser.add_argument("--merged_output_addr", default=f'./rank/LaMP_{T}/train_questions_rank_merge.json', )
    return parser.parse_args()


def main():
    opts = load_args()
    q_addr = opts.lamp_questions_addr
    o_addr = opts.lamp_output_addr  # for train & val
    rank_addr = opts.profile_ranking_addr
    res_addr = opts.merged_output_addr

    with open(q_addr) as fp:
        inp = json.load(fp)
    with open(o_addr) as fp:
        out = json.load(fp)

    if rank_addr:
        with open(rank_addr, 'r') as fp:
            rank = json.load(fp)
    else:
        rank = dict()
        for data in inp:
            rank[data['id']] = []
            for item in data['profile']:
                rank[data['id']].append(item['id'])

    with open(res_addr, "w") as fp:
        res = merge(inp, out, rank)
        json.dump(res, fp, indent=4)

if __name__ == "__main__":
    main()

