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

