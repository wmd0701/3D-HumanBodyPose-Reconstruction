#!/bin/sh

CPU=6
MEM=2048
Q=25:00

cd codebase/ && bsub -n $CPU -W $Q -o out.txt -R "rusage[mem=${MEM}, ngpus_excl_p=1]" python $@
