#!/bin/bash

# Check if number of workers is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 <number_of_workers>"
	echo "Example: $0 4"
	exit 1
fi

NUM_WORKERS=$1

# Validate that the input is a positive integer
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [ "$NUM_WORKERS" -le 0 ]; then
	echo "Error: Number of workers must be a positive integer"
	exit 1
fi

echo "Running experiments with $NUM_WORKERS parallel workers..."

# cd ../reward_machines

# Function to run all experiments for a single iteration
run_iteration() {
	local i=$1
	echo "Starting iteration $i..."
	
	# Multi-task
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql/Office-random/M1/$i
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql-rs/Office-random/M1/$i --use_rs
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm/Office-random/M1/$i --use_crm
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm-rs/Office-random/M1/$i --use_crm --use_rs
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=hrm --env=Office-random-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm/Office-random/M1/$i
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=hrm --env=Office-random-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm-rs/Office-random/M1/$i --use_rs

	# Single task
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql/Office-random-single/M1/$i
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/ql-rs/Office-random-single/M1/$i --use_rs
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm/Office-random-single/M1/$i --use_crm
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=qlearning --env=Office-random-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/crm-rs/Office-random-single/M1/$i --use_crm --use_rs
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=hrm --env=Office-random-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm/Office-random-single/M1/$i
	/home/christopher/miniconda3/envs/hrm/bin/python run.py --alg=hrm --env=Office-random-single-v0 --num_timesteps=1e5 --gamma=0.9 --log_path=../my_results/hrm-rs/Office-random-single/M1/$i --use_rs
	
	echo "Completed iteration $i"
}

# Export the function so it can be used by subshells
export -f run_iteration

# Run iterations in parallel using xargs
seq 0 59 | xargs -n 1 -P $NUM_WORKERS -I {} bash -c 'run_iteration {}'

echo "All experiments completed!"