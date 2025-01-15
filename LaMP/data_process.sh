
# nohup bash data_process.sh >dp1.out 2>&1 &

export CUDA_VISIBLE_DEVICES=0

rankers=("contriever")  # "bm25"  "recency"  需要date字段
#for task_id in 1 2 3 4 5 6 7
#tasks=('IMDB-B' 'YELP-B')
tasks=('GDRD-B' 'PPR-B')
splits=('train' 'dev' 'test')

for task in "${tasks[@]}"
do
    mkdir ./rank/${task}

    for _split in "${splits[@]}"
    do

        for ranker in "${rankers[@]}"
        do

            echo process ${task_id}  ${_split} ${ranker} rank
            python rank_profiles.py \
                --input_data_addr ../data/${task}/${_split}_questions.json \
                --output_ranking_addr ./rank/${task}/${_split}_questions_rank_${ranker}.json \
                --task ${task} \
                --ranker ${ranker} \
                --contriever_checkpoint /data/hf/facebook/contriever

            python merge_with_rank.py \
                --lamp_questions_addr ../data/${task}/${_split}_questions.json \
                --lamp_output_addr ../data/${task}/${_split}_outputs.json \
                --profile_ranking_addr ./rank/${task}/${_split}_questions_rank_${ranker}.json \
                --merged_output_addr ./rank/${task}/${_split}_questions_rank_${ranker}_merge.json
        done
    done
done

#### time split

# nohup bash data_process.sh >dp2.out 2>&1 &

#export CUDA_VISIBLE_DEVICES=1
#rankers=("contriever")  # "bm25" “contriever”  "recency"  需要date字段
##for task_id in 1 3 4 5 7
#for task in "${tasks[@]}"
#do
#    # task=LaMP_t_${task_id}
#    mkdir ./rank/${task}
#
#    for _split in train dev
#    do
#
#        for ranker in "${rankers[@]}"
#        do
#
#            echo process ${task}  ${_split} ${ranker} rank
#
#            python rank_profiles.py \
#                --input_data_addr ../data/${task}/${_split}_questions.json \
#                --output_ranking_addr ./rank/${task}/${_split}_questions_rank_${ranker}.json \
#                --task  ${task} \
#                --ranker ${ranker} \
#                --contriever_checkpoint /data/hf/facebook/contriever \
#                --use_date
#
#            python merge_with_rank.py \
#                --lamp_questions_addr ../data/${task}/${_split}_questions.json \
#                --lamp_output_addr ../data/${task}/${_split}_outputs.json \
#                --profile_ranking_addr ./rank/${task}/${_split}_questions_rank_${ranker}.json \
#                --merged_output_addr ./rank/${task}/${_split}_questions_rank_${ranker}_merge.json
#        done
#    done
#done