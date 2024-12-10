#!/bin/bash --login

# normally has 40 G, 35 minutes, 1 gpu, 4 cpus

module purge
module load ffmpeg/7.0.1-3nxkwho
source /home/someUser/.bashrc
conda activate garf

srun -u echo "y" | python evaluate.py --model=garf --yaml=fineview --group=butterfly2 --name=fineview_gauss2 --output_root="./logs" --data.dataset=fineview --data.scene="butterfly" --optim.sched=! --init.pose=True --camera.novel.topCam=True --data.preshuffle=True --arch.gausssian.sigma=0.02

