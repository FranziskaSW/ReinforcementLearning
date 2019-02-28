#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/usr/franziska/ReinforcementLearning/logs/
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);Linear(epsilon=0.1);Linear(epsilon=0.2);Linear(epsilon=0.3);Linear(epsilon=0.4)" \
    -D 5000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 15 \
    -l linear_01_eps.log \
    -o linear_01_eps.out \
    -rt linear_01_eps.pickle

