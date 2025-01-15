
# nohup bash mcts.sh >mcts.out 2>&1 &

rankers=("contriever")
tasks=('LaMP_3')
splits=('dev')
for task in "${tasks[@]}"
do
  for _split in "${splits[@]}"
  do
      echo process ${task}
      for ranker in "${rankers[@]}"
      do

      python mcts.py \
          --task ${task} \
          --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct \
          --retriever ${ranker} \
          --use_profile \
          --is_ranked \
          --max_length 1024 \
          --method MCTS \
          --split ${_split}
      done
  done
done
