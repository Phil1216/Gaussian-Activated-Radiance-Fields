#!/bin/bash --login


#normally has 50 gigs mem and 24 hours, 1 gpu, 4 cpus

#mamba activate nerf-pytorch-new

module purge
module load ffmpeg/7.0.1-3nxkwho
source /home/someUser/.bashrc
conda activate garf

srun -u echo "y" | python train.py --model=garf --yaml=fineview --group=butterfly2 --name=fineview_gauss2 --output_root="./logs" --data.dataset=fineview --data.scene="butterfly" --optim.sched=! --init.pose=True --init.pose_warmup=2000 --optim.lr_pose=0.0002 --data.preshuffle=True --arch.gausssian.sigma=0.02

