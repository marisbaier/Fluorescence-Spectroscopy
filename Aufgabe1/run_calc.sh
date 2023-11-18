#!/bin/bash

#SBATCH --job-name=semesterprojectKI
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100

jupyter nbconvert --to script Aufgabe1.ipynb 

module load cuda

python3 Aufgabe1.py