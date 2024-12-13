SCENE_DIR="data/zipnerf_undistorted"
SCENE_LIST="berlin london nyc alameda"
DATA_FACTOR=4
RENDER_TRAJ_PATH="ellipse"

RESULT_DIR="results/benchmark_mcmc_2M_zipnerf_undistorted"
CAP_MAX=2000000

# RESULT_DIR="results/benchmark_mcmc_4M_zipnerf_undistorted"
# CAP_MAX=4000000

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --opacity_reg 0.001 \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --camera_model pinhole \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done
