#!/bin/bash
#cd ../reward_machines
python3 run.py --env=diag3x3-sparse-v0 --num_timesteps=3e6 --gamma=0.99 --alg=dhrm --log_path=../my_results/dense3x3J -r_max=100
