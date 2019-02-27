#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --output=<YOUR_OUTPUT_PATH_HERE>
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py -P "Avoid(epsilon=0.5);Avoid(epsilon=0.2);MyPolicy(learning_rate=0.01);MyPolicy(learning_rate=0.001);MyPolicy(learning_rate=0.0001)" -D 7000 -s 1000 -l "/logs/match_01.log" -r 0 -plt 0.01 -pat 0.005 -pit 15

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py -P "Avoid(epsilon=0.5);Avoid(epsilon=0.2);MyPolicy(epsilon=0.3);MyPolicy(epsilon=0.3);MyPolicy(epsilon=0.4)" -D 7000 -s 1000 -l "/logs/match_02.log" -r 0 -plt 0.01 -pat 0.005 -pit 15
