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
from fineview_utils.fineview_directory import FineviewDirectory


class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        super().__init__(opt,split)

        # TODO move to parameter
        speciesIndex = 0
        bd_factor=.75
        crop = True
        factor = 1

        self.root = opt.data.root or "data/fineview"
        self.path = "{}/{}".format(self.root,opt.data.scene)

        self.fineViewDir = FineviewDirectory(self.path, speciesIndex, crop)
        images, poses, bds, K = self.parsePoses(bd_factor, crop, factor)

        print('Data:')
        print(poses.shape, images.shape, bds.shape)
        
        images = images.astype(np.float32)
        poses = poses.astype(np.float32)

        # return images, poses, bds, K

        # self.path_image = "{}/images".format(self.path)
        # image_fnames = sorted(os.listdir(self.path_image))
        # poses_raw,bounds = self.parse_cameras_and_bounds(opt)
        # self.list = list(zip(image_fnames,poses_raw,bounds))
        # # manually split train/val subsets
        # num_val_split = int(len(self)*opt.data.val_ratio)
        # self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        # if subset: self.list = self.list[:subset]
        # # preload dataset
        # if opt.data.preload:
            # self.images = self.preload_threading(opt,self.get_image)
            # self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def parsePoses(self, bd_factor=.75, crop = True, factor = 1):

        testskip = 1
        if testskip==0:
            skip = 1
        else:
            skip = testskip

        imgs = []
        imgs_mask = []
        cam_mats = []
        K = []

        f = h5py.File(self.fineViewDir.camera_param_path, 'r')

        H, W, focal = self.getHWF(f, self.fineViewDir.img_list, crop, factor)        
        hwf = np.array([H,W,focal]).reshape([3,1])
        
        for i in self.fineViewDir.img_list[::skip]:
            
            i_path = Path(i)
            image_original, image_blank = self.initializeMask(i, self.fineViewDir.img_list, i_path, crop)            
            imgs.append(image_original)
            imgs_mask.append(image_blank)
            
            camMat, k_param = self.parseCamParams(i, f, self.fineViewDir.x_min, self.fineViewDir.y_min)
            K.append(k_param)
            cam_mats.append(camMat)

        f.close()

        K = np.stack(K)
        K = K/factor
        cam_mats = np.stack(cam_mats, 0)

        poses = cam_mats[:, :3, :4].transpose([1,2,0])
        #fineview pose is world to camera pose and it is same with opencv coordinate. Convert from (right, down, forward) to (right, up, backward) and change to camera to world coordinate 
        #must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
        poses = np.concatenate([poses[1:2, :, :], poses[0:1, :, :], -poses[2:3, :, :], poses[3:4, :, :]], 1)   


        bds = self.calcBoundaries(self.fineViewDir.sp_folder, poses)

        #render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        imgs_mask = (np.array(imgs_mask) / 255.).astype(np.float32) 

        if factor != 1:
            imgs_resized = np.zeros((imgs.shape[0], H, W, 4))
            for i in range(imgs.shape[0]):
                imgs_resized[i][:,:,0:3] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
                imgs_resized[i][:,:,3] = cv2.resize(imgs_mask[i], (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_resized
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[1:2, :, :], -poses[0:1, :, :], poses[2:, :, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        #no need this 
        #imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        images = imgs
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
        poses[:,:3,3] *= sc
        bds *= sc

        return images, poses, bds, K
    
    def getHWF(self, paramFile, img_list, crop = True, factor = 1):
        if crop:    
            H_original = self.fineViewDir.crop_image_size[1]
            W_original = self.fineViewDir.crop_image_size[0]
        else:
            H_original, W_original = imageio.imread(img_list[0]).shape[:2]

        fx = []
        fy = []
        for i in ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']:
            fx.append(paramFile[i]['mtx'][0,0])
            fy.append(paramFile[i]['mtx'][1,1])
        focals = fx + fy
        focal = np.array(focals).mean() 
        
        H = H_original//factor
        W = W_original//factor
        focal = focal/factor
        return H, W, focal

    def initializeMask(self, imageFile, img_list, img_path, crop = True):

        non_debug = True

        crop_image_size = self.fineViewDir.crop_image_size
        x_min = self.fineViewDir.x_min
        x_max = self.fineViewDir.x_max
        y_min = self.fineViewDir.y_min
        y_max = self.fineViewDir.y_max

        if crop:    
            H_original = crop_image_size[1]
            W_original = crop_image_size[0]
        else:
            H_original, W_original = imageio.imread(img_list[0]).shape[:2]
        
        d_type = imageio.imread(img_list[0]).dtype

        mask_path = Path(self.path).joinpath('crop_mask_undistort', img_path.parts[-4], img_path.parts[-2], img_path.stem + "_mask.png")
        if non_debug:
            image_original = imageio.imread(imageFile)
            
            if crop:
                image_blank = imageio.imread(mask_path)[:,:,0]
            else:
                image_blank = np.zeros((H_original,W_original), dtype=d_type)
                image_blank[y_min:y_max,x_min:x_max] = imageio.imread(mask_path)[:,:,0]
        else:
            image_original = np.empty((100,100,3))
            image_blank = np.empty((100,100,1))

        return image_original, image_blank
    
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

        zvals = np.sum(-(xyz[:, np.newaxis, :].transpose([2,0,1]) - poses[3:4, :3, :]) * poses[2:3, :3, :], 0)
        print( 'Depth stats', zvals.min(), zvals.max(), zvals.mean() )
        
        bds = []
        for i in range(poses.shape[2]):
            zs = zvals[:, i]
            close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
            # print( i, close_depth, inf_depth )
            
            bds.append(np.array([close_depth, inf_depth]))
        return np.array(bds).T

    # important
    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    # essential
    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all

    # superclass (essential)
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

    # superclass (unused?)
    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    # interface, util_vis.py
    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose = self.list[idx][1]
        #pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    # def parse_raw_camera(self,opt,pose_raw):
    #     pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
    #     pose = camera.pose.compose([pose_flip,pose_raw[:3]])
    #     pose = camera.pose.invert(pose)
    #     pose = camera.pose.compose([pose_flip,pose])
    #     return pose
