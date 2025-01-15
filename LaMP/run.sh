

# nohup bash run.sh >sft.out 2>&1 &
# nohup bash run.sh >sbp.out 2>&1 &

export CUDA_VISIBLE_DEVICES=1
#rankers=("bm25" "contriever" "recency")
rankers=("contriever")
#for task_id in 1 2 3 4 5 6 7  # 145 7
for task_id in 1
do
    echo process ${task_id}
    for ranker in "${rankers[@]}"
    do

#    python train_llm.py \
#        --method SFT \
#        --peft_type LORA \
#        --task LaMP_${task_id} \
#        --model_name /data/hf/google/flan-t5-base \
#        --retriever ${ranker} \
#        --use_profile \
#        --is_ranked \
#        --num_retrieved 1
#    done

    python sbp.py \
        --task LaMP_${task_id} \
        --model_name /data/hf/meta-llama/Meta-Llama-3-8B-Instruct \
        --retriever ${ranker} \
        --use_profile \
        --is_ranked \
        --max_length 1024 \
        --method SBP \
        --split train

    done


#    python evaluate_llm.py \
#        --validation_data /*address to sorted validation data using the previous step*/ \
#        --model_addr /*address to the model that should be used for initialization of the LLM*/ \
#        --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
#        --output_dir /*output directory to save results */ \
#        --use_profile \ /*used to perfrom personalization with RAG */
#        --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
#        --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
#        --num_retrieved /*number of items to be retrieved from the user profile*/
done
