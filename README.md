# 3D Human Body Pose Reconstruction

This repo is for the course Machine Perception, ETHZ 2021 autumn semester. The goal is to reconstruct 3D human body pose from 2D images. Our model consists of three sequential parts: a convolutional encoder, an iterative regressor and a SMPL human body model.

Detailed task description can be found on the ETHZ AIT [project plattform](https://machine-perception.ait.ethz.ch/). All the data and a SMPL framework are already given and stored on ETHZ [Leonhard](https://scicomp.ethz.ch/wiki/Leonhard) cluster. For access to both plattform and cluster, a VPN connection is needed if not accessed from inside ETHZ eduroam.

**Important**: As ETHZ merged [Euler cluster](https://scicomp.ethz.ch/wiki/Euler) and Leonhard cluster, the data on Leonhard is lost. For reproducing the experiment results, please contact ETHZ AIT group for project framework and data.

## How to train the final model

1. Activate Enviroment
```
conda activate mp_project3
```
2. Submit training job
```
bash train.sh
```

## How to predict with the final model

1. Activate Enviroment
```
conda activate mp_project3
```
2. Make predictions with the last model. Look in the out directory for the last iteration number (should be 40000). Use that model to make predictions
```
cd codebase/ && bsub -n 6 -W 00:30 -o out.txt -R "rusage[mem=4096, ngpus_excl_p=1]" python test.py ../configs/modelfinal.yaml --model_file model_40000.pt
```

