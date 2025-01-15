

# nohup bash sbp.sh >sbp.out 2>&1 &

export CUDA_VISIBLE_DEVICES=0
#rankers=("bm25" "contriever" "recency")
rankers=("contriever")
#splits=('train' 'dev' 'test')
#splits=('train' 'dev')
splits=('dev')

for task_id in 3
#tasks=('IMDB-B' 'YELP-B')
#tasks=('GDRD-B' 'PPR-B')
#for task in "${tasks[@]}"
do
  task=LaMP_t_${task_id}
  for _split in "${splits[@]}"
  do
      echo process ${task}
      for ranker in "${rankers[@]}"
      do

      python sbp.py \
          --task ${task} \
          --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct \
          --retriever ${ranker} \
          --use_profile \
          --is_ranked \
          --max_length 1024 \
          --method SBP \
          --split ${_split}
      done
  done
done
