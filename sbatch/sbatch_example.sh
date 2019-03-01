#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/usr/franziska/ReinforcementLearning/logs
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/Documents/ReinforcementLearning/Snake.py \
    -P "Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);Avoid(epsilon=0.1);MyPolicy(batch_size=8)" \
    -D 20000 -s 2000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -l "Q_bs_8.log" \
    -o "Q_bs_8.out"