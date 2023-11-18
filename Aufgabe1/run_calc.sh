#!/bin/bash

#SBATCH --job-name=semesterprojectKI
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100

module load cuda

python3 Aufgabe1.py