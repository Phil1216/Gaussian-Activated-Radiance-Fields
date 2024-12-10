#!/bin/bash --login

# Normally has 40G, 8 minutes, 1 gpu, 4 cpus

module purge
module load ffmpeg/7.0.1-3nxkwho
source /home/someUser/.bashrc
conda activate garf

srun -u echo "y" | python evaluate.py --model=garf --yaml=garf_llff --group=0_test --name=fern --output_root="./logs" --data.dataset=llff --data.scene="fern" --optim.sched=! --init.pose=True --camera.novel.topCam=True --camera.novel.spiral=True

