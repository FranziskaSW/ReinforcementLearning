#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=<YOUR_OUTPUT_PATH_HERE>
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/shais/dsgissin/apml_snake/bin/activate.csh
module load tensorflow

python3 /cs/usr/franziska/Documents/ReinforcementLearning/TestSnake.py \
    -D 7000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.log" \
    -o "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.out" \
    -rt match_01.pickle \
    -ds _Q_3 -tp epsilon -ss 3

python3 /cs/usr/franziska/Documents/ReinforcementLearning/TestSnake.py \
    -D 7000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.log" \
    -o "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.out" \
    -rt match_01.pickle \
    -ds _Q_3 -tp learning_rate -ss 3

python3 /cs/usr/franziska/Documents/ReinforcementLearning/TestSnake.py \
    -D 7000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.log" \
    -o "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.out" \
    -rt match_01.pickle \
    -ds _Q_3 -tp gamma -ss 3


python3 /cs/usr/franziska/Documents/ReinforcementLearning/TestSnake.py \
    -D 7000 -s 1000 -r 0 -plt 0.01 -pat 0.005 -pit 60 \
    -l "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.log" \
    -o "/cs/usr/franziska/Documents/ReinforcementLearning/logs/match_01.out" \
    -rt match_01.pickle \
    -ds _Q_3 -tp batch_size -ss 3
