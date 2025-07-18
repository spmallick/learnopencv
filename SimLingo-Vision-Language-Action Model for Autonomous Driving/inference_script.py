! export CARLA_ROOT=/path/to/CARLA/root
! export WORK_DIR=/path/to/simlingo
! export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
! export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
! export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
! export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
! 
! git clone https://github.com/RenzKa/simlingo.git
! cd simlingoy./setup_carla.sh
! conda env create -f environment.yaml
! conda activate simlingo
! 
! bash ./start_eval_simlingo.py --mode closed_loop