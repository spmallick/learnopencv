
device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train.py --dataset thyroid --train_data_path /home/shubham/Work/AnomalyCLIP/Thyroid_Dataset/tn3k \
        --save_path ${save_dir} \
        --features_list 24 --image_size 518  --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done


