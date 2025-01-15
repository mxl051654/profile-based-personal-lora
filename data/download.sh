
# 并行版本  加 &
# nohup bash download.sh &


# user based data split
#for task_id in 1 2 3 4 5 6 7
#do
#    mkdir LaMP_${task_id}
#    cd LaMP_${task_id}
#    for _split in train dev
#    do
#        wget https://ciir.cs.umass.edu/downloads/LaMP/LaMP_${task_id}/${_split}/${_split}_questions.json &
#        wget https://ciir.cs.umass.edu/downloads/LaMP/LaMP_${task_id}/${_split}/${_split}_outputs.json &
#    done
#    wget https://ciir.cs.umass.edu/downloads/LaMP/LaMP_${task_id}/test/test_questions.json &
#    cd ../
#done
#
#wait  # 等待所有后台进程完成


# time based data split
#for task_id in 1 2 3 4 5 6 7
for task_id in 5
do
    mkdir LaMP_t_${task_id}  # LaMP_t_1
    cd LaMP_t_${task_id}

    for _split in train dev
    do
        wget https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_${task_id}/${_split}/${_split}_questions.json &
        wget https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_${task_id}/${_split}/${_split}_outputs.json &
    done
    wget https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_${task_id}/test/test_questions.json &

    cd ../
done

wait  # 等待所有后台进程完成