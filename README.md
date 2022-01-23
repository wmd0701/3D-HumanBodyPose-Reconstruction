# 3D Human Body Pose Reconstruction

This repo is for the course Machine Perception, ETHZ 2021 autumn semester and imported from [ETHZ INFK GitLab](https://gitlab.inf.ethz.ch/COURSE-MP2021/AMA21). The goal is to reconstruct 3D human body pose from 2D images. Detailed task description can be found on the ETHZ AIT [project plattform](https://machine-perception.ait.ethz.ch/). All the data and a SMPL framework are already given and stored on ETHZ [Leonhard](https://scicomp.ethz.ch/wiki/Leonhard) cluster. For access to both plattform and cluster, a VPN connection is needed if not accessed from inside ETHZ eduroam.

Our model consists of three sequential parts: a convolutional encoder, an iterative regressor and a SMPL human body model.

**Important:** As ETHZ merged [Euler](https://scicomp.ethz.ch/wiki/Euler) cluster and Leonhard cluster, the data on Leonhard is lost. For reproducing the experiment results, please contact ETHZ AIT group for project framework and data.

## Visualizations

### Pipeline
<img src="https://user-images.githubusercontent.com/34072813/150693234-0a5ab807-9c51-4de9-bc1a-ffa5fbb27135.png" width=40% height=40%>

### Iterative regressor
<img src="https://user-images.githubusercontent.com/34072813/150693242-3866990c-c16b-428a-b5d8-a18549c23125.PNG" width=25% height=25%>

### Mesh reconstruction
<img src="https://user-images.githubusercontent.com/34072813/150693250-5ac0f6da-6953-432d-869d-ee551b6a3c71.PNG" width=25% height=25%>



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

