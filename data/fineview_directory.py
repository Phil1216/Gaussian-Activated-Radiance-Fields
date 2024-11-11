import os
import glob
import h5py
import imageio

class FineviewDirectory():

    def __init__(self, path, speciesIndex = 0, crop = True, factor = 1):
        self.path = path
        self.speciesIndex = speciesIndex
        self.crop = crop
        self.factor = factor

        self.crop_param_path = self.path + '/crop_pram_undistort.h5'
        self.camera_param_path = self.path + '/camera_pram_2_no180_2_opt.h5'
        
        self.x_min, self.x_max, self.y_min, self.y_max, self.crop_image_size = self.getCropParams()
        self.img_list, self.speciesFolder = self.gatherImages()

        tmp = imageio.imread(self.img_list[0])
        self.image_file_size = tmp.shape[:2]
        self.file_d_type = tmp.dtype

    def getCropParams(self):
        fc = h5py.File(self.crop_param_path, 'r')
        x_min,y_min = fc["crop/offset"][self.speciesIndex]
        crop_image_size = fc["crop/img_size"][self.speciesIndex]
        x_max = crop_image_size[0] + x_min
        y_max = crop_image_size[1] + y_min
        fc.close()

        return x_min, x_max, y_min, y_max, crop_image_size

    def gatherImages(self):

        if self.crop:
            extention = 'png'
            img_folder ="crop_undistort"
        else:
            extention = 'JPG'
            img_folder ="undistort"

        folder_list = glob.glob(self.path + "/" + img_folder + "/*")
        folder_list.sort()

        l1 = glob.glob(folder_list[self.speciesIndex] + '/images/camera1/*.' + extention)
        l2 = glob.glob(folder_list[self.speciesIndex] + '/images/camera2/*.' + extention)
        l3 = glob.glob(folder_list[self.speciesIndex] + '/images/camera3/*.' + extention)
        l4 = glob.glob(folder_list[self.speciesIndex] + '/images/camera4/*.' + extention)
        l5 = glob.glob(folder_list[self.speciesIndex] + '/images/camera5/*.' + extention)
        l6 = glob.glob(folder_list[self.speciesIndex] + '/images/camera6/*.' + extention)
        l7 = glob.glob(folder_list[self.speciesIndex] + '/images/camera7/*.' + extention)
        l8 = glob.glob(folder_list[self.speciesIndex] + '/images/camera8/*.' + extention)
        l1.sort()
        l2.sort()
        l3.sort()
        l4.sort()
        l5.sort()
        l6.sort()
        l7.sort()
        l8.sort()

        img_list = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
        img_list.sort()

        speciesFolder = os.path.basename(folder_list[self.speciesIndex])

        return img_list, speciesFolder