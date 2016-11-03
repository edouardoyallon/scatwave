for N in 500 1000 2000 4000 8000
do
for i in 1 2 3 4 5
do
th train_cifar10_small_dataset.lua --N $N --iter $i --save "log_cr10_smll_data_${N}_${i}";
done
done


