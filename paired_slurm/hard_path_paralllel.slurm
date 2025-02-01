#!/usr/bin/env bash
#SBATCH --mem-per-gpu=24G
#SBATCH --cpus-per-gpu=8
#SBATCH --open-mode=append
#SBATCH --job-name=pdirl
#SBATCH --gres=gpu:1
#SBATCH --array=0



#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --time=12:00:00
TIMEOUT=12h


/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/envs/new-hrm/bin/wandb login
echo finished wandb login

N_JOB=1
FULL_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo the full id is $FULL_ID

EXPERIMENT=pdirl
ENV=diag3x3-dense-v0
WANDB_ENTITY=penn-pal
WANDB_NAME=pointmaze_slurm_serial
WANDB_TAG=hrm-diag3x3

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES

EXP_NAME=$EXPERIMENT-$ENV-$FULL_ID
OUT_DIR=outputs/$EXP_NAME


# Added by chris:                                                                                                                                                                             
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ccwatson/.mujoco/mujoco210/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chriswatson/.mujoco/mujoco210/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/chriswatson/.mujoco/mujoco210/bin
# Not sure if this is good
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chriswatson/.mujoco/mujoco200/bin

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate new-hrm
# Function to find an available port in the range 5000-9999
find_free_port() {
    while :; do
        PORT=$(( RANDOM % 5000 + 5000 ))  # Generate a port between 5000-9999
        if ! lsof -i :$PORT >/dev/null; then
            echo $PORT
            return
        fi
    done
}

echo "About to start running"

declare -a SERVER_PIDS  # Array to store server PIDs

for ((i=0; i<$N_JOB; i++))
do
    TASK_OUT_DIR=$OUT_DIR/task$i
    TASK_LOG_DIR=$TASK_OUT_DIR/logs

    mkdir -p $TASK_LOG_DIR

    scontrol show -dd job $SLURM_JOB_ID > $TASK_LOG_DIR/slurm.out 2>&1
    printenv >> $TASK_LOG_DIR/slurm.out 2>&1
    scontrol write batch_script $SLURM_JOB_ID $TASK_LOG_DIR/sbatch.slurm >> $TASK_LOG_DIR/sbatch.slurm 2>&1
    git rev-parse HEAD >> $TASK_LOG_DIR/head_commit_hash

    # Get a free port
    FREE_PORT=$(find_free_port)
    echo "Random available port: $FREE_PORT"

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chriswatson/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES=0
    export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES

    # Start the server in the background and capture its PID
    /mnt/kostas-graid/sw/envs/chriswatson/miniconda3/envs/jax-conda-new/bin/python3 ../../pdirl/server.py $FREE_PORT >$TASK_LOG_DIR/server.out 2>$TASK_LOG_DIR/server.err &
    SERVER_PIDS[$i]=$!  # Store the server PID in the array

    echo "Server started (PID: ${SERVER_PIDS[$i]}) on port $FREE_PORT"

    sleep 10  # Allow time for the server to initialize

    export PYOPENGL_PLATFORM=egl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chriswatson/.mujoco/mujoco210/bin

    # Run the training script in parallel
    /mnt/kostas-graid/sw/envs/chriswatson/miniconda3/envs/new-hrm/bin/python3 -u run.py \
        --vram_frac=0.2 \
        --wandb_experiment=$EXPERIMENT \
        --port=$FREE_PORT \
        --wandb_name=$WANDB_NAME \
        --wandb_entity=$WANDB_ENTITY \
        --env=$ENV \
        --wandb_tag=$WANDB_TAG \
        --num_timesteps=10e6 \
        --gamma=0.99 \
        --alg=dhrm \
        --log_path=$TASK_LOG_DIR \
        -r_max=100 \
        >> $TASK_LOG_DIR/app.log 2>&1 &  # Background run.py as well

    TRAIN_PIDS[$i]=$!  # Store the training script PID

done

# Function to clean up all running servers
cleanup() {
    echo "Shutting down all server processes..."
    for pid in "${SERVER_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            echo "Stopping server (PID: $pid)"
            kill $pid
            sleep 3  # Allow graceful termination
            if ps -p $pid > /dev/null; then
                echo "Force killing server process $pid..."
                kill -9 $pid
            fi
        fi
    done
}

# Trap SIGTERM, SIGINT, and SLURM job exit to clean up servers
trap cleanup SIGTERM SIGINT EXIT

# Wait for all training jobs to finish
wait "${TRAIN_PIDS[@]}"

# Clean up servers when all jobs are done
cleanup

echo "All jobs completed."
exit 0
