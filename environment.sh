#!/bin/bash
conda create --yes --name bayes-particle python=3.8 pybamm numpy scipy matplotlib
source ~/anaconda3/etc/profile.d/conda.sh
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate bayes-particle
pip install -e .
