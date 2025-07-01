device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do

    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_thyroid
        save_dir=./checkpoints/${base_dir}/
        echo ${base_dir}
        echo ${save_dir}
        echo ${depth}
        echo ${n_ctx}
        echo ${t_n_ctx}
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset thyroid \
        --data_path /home/shubham/Work/AnomalyCLIP/Thyroid_Dataset/tn3k --save_path /home/shubham/Work/AnomalyCLIP/results/singlescale_tn3k \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done
