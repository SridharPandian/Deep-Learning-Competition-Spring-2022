#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2022sp_15

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=10:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

#cp -rp /scratch/DL22SP/unlabeled_224.sqsh /tmp
#cp -rp /scratch/DL22SP/unlabeled_224.sqsh /scratch/$USER/
cp -rp /scratch/DL22SP/labeled.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
--overlay /tmp/labeled.sqsh \
--overlay /scratch/ss14499/unlabeled_224.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate
python train.py -o resume_finetune99 --backbone_path dino-bb/checkpoint0099.pth -e dino-resume_finetune99 -r dino-weights/dino_99_finetuned_8.pth --backbone_type dino
"
