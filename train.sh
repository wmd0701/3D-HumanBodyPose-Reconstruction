#!/bin/sh

CPU=6
MEM=4096
Q=96:00

# Load eth proxy
module load eth_proxy

# submit train job
cd codebase/ && bsub -n $CPU -W $Q -o out.txt -R "rusage[mem=${MEM}, ngpus_excl_p=1]" python train.py ../configs/modelfinal.yaml
