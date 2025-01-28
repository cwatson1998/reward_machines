#!/bin/bash
#cd ../reward_machines
python3 run.py --alg=dhrm --env=diag3x3-dense-v0 --num_timesteps=3e6 --gamma=0.99 --log_path=./my_results/diag3x3
