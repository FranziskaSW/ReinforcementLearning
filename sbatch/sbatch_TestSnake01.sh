#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=/cs/usr/franziska/ReinforcementLearning/
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/ReinforcementLearning/TestSnake.py \
    -D 10000 -s 2000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -ds _Q_3 -tp epsilon -ss 3

python3 /cs/usr/franziska//ReinforcementLearning/TestSnake.py \
    -D 10000 -s 2000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -ds _Q_3 -tp learning_rate -ss 3

python3 /cs/usr/franziska//ReinforcementLearning/TestSnake.py \
    -D 10000 -s 2000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -ds _Q_3 -tp gamma -ss 3

python3 /cs/usr/franziska//ReinforcementLearning/TestSnake.py \
    -D 10000 -s 2000 -r 0 -plt 0.05 -pat 0.01 -pit 60 \
    -ds _Q_3 -tp batch_size -ss 3
