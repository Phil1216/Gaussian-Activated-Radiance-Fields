import numpy as np
import os
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import glob
import h5py
import cv2
from pathlib import Path
import open3d as o3d
from . import base
import camera
from util import log,debug
from data.fineview_directory import FineviewDirectory


class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 3377,3568
        super().__init__(opt,split)

        # TODO move to parameter
        speciesIndex = 0
        bd_factor=.75
        crop = True
        factor = 1
        
        self.root = opt.data.root or "data/fineview"
        self.path = "{}/{}".format(self.root,opt.data.scene)

        self.fineViewDir = FineviewDirectory(self.path, speciesIndex, crop, factor)
        poses_raw, bds, K = self.parsePoses(bd_factor)

        print('Data:')
        print(poses_raw.shape, bds.shape)

        self.list = list(zip(self.fineViewDir.img_list, poses_raw, bds, K))

        # manually split train/val subsets
        num_val_split = int(len(self)*opt.data.val_ratio)
        self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        if subset: self.list = self.list[:subset]

        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def parsePoses(self, bd_factor=.75):
        factor = self.fineViewDir.factor

        cam_mats = []
        K = []

        f = h5py.File(self.fineViewDir.camera_param_path, 'r')

        H, W, focal = self.getHWF(f)
        assert(self.raw_H==H and self.raw_W==W)

        self.focal = focal/factor
        self.raw_W = W//factor
        self.raw_H = H//factor
        
        for i in self.fineViewDir.img_list:
            
            camMat, k_param = self.parseCamParams(i, f, self.fineViewDir.x_min, self.fineViewDir.y_min)
            K.append(k_param)
            cam_mats.append(camMat)

        f.close()

        K = np.stack(K)
        K = K/factor
        cam_mats = np.stack(cam_mats, 0)
        c2w_mats = np.linalg.inv(cam_mats)

        poses = c2w_mats[:, :3, :4].transpose([1,2,0])
        #fineview pose is world to camera pose and it is same with opencv coordinate. Convert from (right, down, forward) to (right, up, backward) and change to camera to world coordinate 
        #must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t] (ie we start from [r, -u, t] and not from [r, u, -t])
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

        bds = self.calcBoundaries(self.fineViewDir.speciesFolder, poses)

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        
        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
        poses[:,:3,3] *= sc
        bds *= sc

        # remove last column that contains hwf since it's not necessary
        poses = poses[:, :, :4]

        poses = poses.astype(np.float32)
        bds = bds.astype(np.float32)
        K = K.astype(np.float32)

        poses = torch.from_numpy(poses)
        bds = torch.from_numpy(bds)
        K = torch.from_numpy(K)

        return poses, bds, K
    
    def getHWF(self, paramFile):
        if self.fineViewDir.crop:    
            H_original = self.fineViewDir.crop_image_size[1]
            W_original = self.fineViewDir.crop_image_size[0]
        else:
            H_original, W_original = self.fineViewDir.image_file_size

        fx = []
        fy = []
        for i in ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']:
            fx.append(paramFile[i]['mtx'][0,0])
            fy.append(paramFile[i]['mtx'][1,1])
        focals = fx + fy
        focal = np.array(focals).mean() 

        return H_original, W_original, focal
    
    def parseCamParams(self, imageFile, paramFile, x_min, y_min):
        #pose conversion from fineview data
        camera = imageFile[-14:-7]
        i_number = int(imageFile[-6:-4])
        r_vec = paramFile[camera]['rvec'][i_number]
        t_vec = paramFile[camera]['tvec'][i_number]

        k_param = paramFile[camera]['mtx'][:]
        k_param[0,2] -= x_min
        k_param[1,2] -= y_min

        mat = np.concatenate([r_vec, t_vec],  axis=1)
        tmp = np.array([0,0,0,1])
        mat4 = np.vstack((mat, tmp.T))

        #[[R R R t]
        # [R R R t]
        # [R R R t]
        # [0 0 0 1]]
        return mat4, k_param

    def calcBoundaries(self, sp_folder, poses):
        pc_path = self.path + "/correspondence/" + sp_folder + "/" + sp_folder + ".pcd"  
        pcd = o3d.io.read_point_cloud(pc_path)
        xyz = np.asarray(pcd.points)      

        zvals = np.sum(-(xyz[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
        print( 'Depth stats', zvals.min(), zvals.max(), zvals.mean() )
        
        bds = []
        for i in range(poses.shape[2]):
            zs = zvals[:, i]
            close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
            # print( i, close_depth, inf_depth )
            
            bds.append(np.array([close_depth, inf_depth]))
        return np.array(bds).T

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)

        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = self.fineViewDir.img_list[idx]
        img = self.loadMaskedImg(image_fname)

        return PIL.Image.fromarray(img)
    
    def loadMaskedImg(self, imageFile):
        factor = self.fineViewDir.factor
        x_min = self.fineViewDir.x_min
        x_max = self.fineViewDir.x_max
        y_min = self.fineViewDir.y_min
        y_max = self.fineViewDir.y_max

        img_path = Path(imageFile)
        mask_path = Path(self.path).joinpath('crop_mask_undistort', img_path.parts[-4], img_path.parts[-2], img_path.stem + "_mask.png")
        
        image_original = imageio.imread(imageFile)            
        if self.fineViewDir.crop:
            image_mask = imageio.imread(mask_path)[:,:,0]
        else:
            image_mask = np.zeros(self.fineViewDir.image_file_size, dtype=self.fineViewDir.file_d_type)
            image_mask[y_min:y_max,x_min:x_max] = imageio.imread(mask_path)[:,:,0]


        # Note: PIL doesn't like division by 255 and produces the error "KeyError: ((1, 1, 3), '<f4')""
        img = (np.array(image_original) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        mask = (np.array(image_mask) / 255.).astype(np.float32) 

        if factor != 1:
            img_resized = np.zeros((self.raw_H, self.raw_W, 4))
            img_resized[:,:,0:3] = cv2.resize(img, (self.raw_W, self.raw_H), interpolation=cv2.INTER_AREA)
            img_resized[:,:,3] = cv2.resize(mask, (self.raw_W, self.raw_H), interpolation=cv2.INTER_AREA)

            img = img_resized
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        # PIL doesn't like division by 255, so undo it
        img = (img * 255.).astype(np.uint8)
        return img

    def get_camera(self,opt,idx):
        intr = self.list[idx][3]
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        # It was already inverted once, no need to do it again
        # pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        # pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        # pose = camera.pose.invert(pose)
        # pose = camera.pose.compose([pose_flip,pose])

        return pose_raw
