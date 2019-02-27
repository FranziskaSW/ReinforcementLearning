#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=<YOUR_OUTPUT_PATH_HERE>
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.5);Avoid(epsilon=0.2);MyPolicy(learning_rate=0.01);MyPolicy(learning_rate=0.001);MyPolicy(learning_rate=0.0001)" \
    -D 7000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.log" \
    -o "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.out" \
    -rt match_01.pickle

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.5);Avoid(epsilon=0.2);MyPolicy(epsilon=0.2);MyPolicy(epsilon=0.3);MyPolicy(epsilon=0.4)"
    -D 7000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_02.log" \
    -o "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_02.out" \
    -rt match_02.pickle
