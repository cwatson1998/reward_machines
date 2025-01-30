
N_JOB=1
FULL_ID="$RANDOM"
echo the full id is $FULL_ID

EXPERIMENT=pdirl
ENV=diag3x3-dense-v0
WANDB_ENTITY=penn-pal
WANDB_NAME=pointmaze_slurm
WANDB_TAG=delete_this

# export MUJOCO_GL=egl
# export CUDA_VISIBLE_DEVICES=0
# export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES

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
# __conda_setup="$('/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/mnt/kostas-graid/sw/envs/chriswatson/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# <<< conda initialize <<<

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


echo About to start running

for((i=0; i<$N_JOB; i++))
do
    TASK_OUT_DIR=$OUT_DIR/task$i
    TASK_LOG_DIR=$TASK_OUT_DIR/logs

    mkdir -p $TASK_LOG_DIR

    # scontrol show -dd job $SLURM_JOB_ID > $TASK_LOG_DIR/slurm.out 2>&1
    # printenv >> $TASK_LOG_DIR/slurm.out 2>&1

    # scontrol write batch_script $SLURM_JOB_ID $TASK_LOG_DIR/sbatch.slurm >> $TASK_LOG_DIR/sbatch.slurm 2>&1

    git rev-parse HEAD >> $TASK_LOG_DIR/head_commit_hash


        
    # Get a free port
    FREE_PORT=$(find_free_port)

    echo "Random available port: $FREE_PORT"



    
#    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chriswatson/.mujoco/mujoco210/bin
#     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    conda activate pdirl

    # export MUJOCO_GL=egl
    # export CUDA_VISIBLE_DEVICES=0
    # export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES


    python ../../pdirl/server.py $FREE_PORT &
    SERVER_PID=$!
    trap "echo 'SLURM job terminating, killing server (PID: $SERVER_PID)'; kill $SERVER_PID 2>/dev/null" SIGTERM SIGINT EXIT

    sleep 15

    conda activate hrm
    XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u run.py \
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
        --r_max=100 \
        >> $TASK_LOG_DIR/app.log 2>&1
        # Do not send this to the background. It is important for it to block.
        TRAIN_EXIT_CODE=$?

        echo "Shutting down server process (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null
        if ps -p $SERVER_PID > /dev/null; then
            kill -9 $SERVER_PID
        fi

        
done
echo "Shutting down server process (PID: $SERVER_PID)"
if ps -p $SERVER_PID > /dev/null; then
    kill $SERVER_PID
    sleep 3  # Give it time to terminate gracefully
    if ps -p $SERVER_PID > /dev/null; then
        echo "Force killing server process..."
        kill -9 $SERVER_PID
    fi
fi
echo All processes complete!
exit 0

