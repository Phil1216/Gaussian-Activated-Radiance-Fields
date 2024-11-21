import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import options
from data.fineview import Dataset as llffDataset

#from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
#visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [-10, 10])
#visualizer2 = CameraPoseVisualizer([-7, 7], [-7, 7], [-7, 7])
#visualizer2 = CameraPoseVisualizer([-1, 1], [-1, 1], [-1, 1])
#visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-3.75, -4.25])
from camera_visualization import camera_visualizer
visualizer = camera_visualizer()

# breakpoint()

# myOptionsDir = '/home/pr245/projects/butterfly/garf/logs/butterfly/up/options.yaml'
# myOptionsDir = '/home/pr245/projects/butterfly/garf/logs/0_test/fern/options.yaml'
myOptionsDir = "./fineview_options.yaml"

opt = options.load_options(myOptionsDir)

# disable train/test split
# opt.data.val_ratio = 0 
opt.data.preload = False

train_data = llffDataset(opt,split="train",subset=opt.data.train_sub)
poses_train = train_data.get_all_camera_poses(opt).cpu().detach().numpy()

eval_data = llffDataset(opt,split="eval",subset=opt.data.train_sub)
poses_eval = eval_data.get_all_camera_poses(opt).cpu().detach().numpy()


poses = np.concatenate([poses_train, poses_eval], 0)
poses = poses[:,:3,:4]


n_poses = np.zeros((poses.shape[0],4,4))
for count, i in enumerate(poses):
    """
    r = i[0:3,0:3]
    t = i[0:3,3]
    r_n = r.T
    t_n = -r.T @ t
    mat = np.concatenate([r_n, t_n.reshape(3,1)],  axis=1)
    
    tmp = np.array([0,0,0,1])
    mat4 = np.vstack((mat, tmp.T))
    mat_i = np.linalg.inv(i)
    """
    #tmp = np.array([0,0,0,1])
    #i = np.vstack((i, tmp.T))
    #n_poses[count] = i
    #visualizer.extrinsic2pyramid(mat4, "red", 0.2)
    #x_180 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    """
    mat_i = np.linalg.inv(i)
    mat_i = np.concatenate([mat_i[0:1, :], -mat_i[1:2, :], -mat_i[2:3, :], mat_i[3:4, :]], 0)
    mat_i = np.linalg.inv(mat_i)
    """
    tmp = np.array([0,0,0,1])
    mat4 = np.vstack((i, tmp.T))
    mat_i = np.linalg.inv(mat4)

    n_poses[count] = mat_i

    #visualizer2.extrinsic2pyramid(i, "red", 1)
#visualizer.plot_camera_scene(poses,0.5,"red","pose")

#Remove 4 out of every 5 poses
remove = np.arange(0, n_poses.shape[0], 5)
remove = np.concatenate([np.arange(i, i + 4) for i in remove])
# n_poses = np.delete(n_poses, remove, axis = 0)

visualizer.plot_camera_scene(n_poses,0.2,"red","pose")

# visualizer.save("test_b001_up_w2c.png")
visualizer.save("test_fineview.png")
# visualizer.save("test_fern_w2c.png")

visualizer.show()
#visualizer2.show()
