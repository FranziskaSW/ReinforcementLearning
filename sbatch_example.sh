#!/bin/csh

#SBATCH --cpus-per-task=2
SBATCH --output=/cs/usr/franziska/Documents/ReinforcementLearning/logs/
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);MyPolicy(epsilon=0.4);MyPolicy(epsilon=0.5);MyPolicy(epsilon=0.6);MyPolicy(epsilon=0.7)" \
    -D 7000 -s 1000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l match_04_eps.log \
    -o match_04_eps.out \
    -rt match_04_eps.pickle

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);MyPolicy(batch_size=8);MyPolicy(batch_size=16);MyPolicy(batch_size=32);MyPolicy(batch_size=64)" \
    -D 7000 -s 1000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l match_04_bs.log \
    -o match_04_bs.out \
    -rt match_04_bs.pickle


python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);MyPolicy(learning_rate=0.01);MyPolicy(learning_rate=0.001);MyPolicy(learning_rate=0.0001);MyPolicy(learning_rate=0.0001)" \
    -D 7000 -s 1000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l match_04_lr.log \
    -o match_04_lr.out \
    -rt match_04_lr.pickle


python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);MyPolicy(gamma=0.1);MyPolicy(gamma=0.3);MyPolicy(gamma=0.5);MyPolicy(gamma=0.7)" \
    -D 7000 -s 1000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l match_04_ga.log \
    -o match_04_ga.out \
    -rt match_04_ga.pickle
