#   用于calibrMat的读取 输出 并对指定图片对矫正 输出矫正后图片 以及拼接图片
import json
import numpy as np
import os
import re
import cv2
import argparse
import glob


class StereoRectify(object):
    def __init__(self, basenum, filepath):
        self.num=basenum
        self.cal_path = filepath    
        self.lsits=[]
        self.mtx_1=[]
        self.dist_1=[]
        self.mtx_2=[]
        self.dist_2=[]
        self.R_list=[]
        self.T_list=[]
        self.E_list=[]
        self.F_list=[]   

        self.read_calibMat(self.cal_path)
        self.Print_mat()
        self.stereoReMap(self.cal_path)

    def read_calibMat(self, cal_path):
        # 读取json文件
        with open('calibMat_' + self.num[:7] + '.json', 'r') as f:
            calibmat = json.load(f)
        self.mtx_1 = calibmat[self.num[:7]]['mtx_1']
        self.dist_1 = calibmat[self.num[:7]]['dist_1']
        self.mtx_2 = calibmat[self.num[:7]]['mtx_2']
        self.dist_2 = calibmat[self.num[:7]]['dist_2']
        self.R_list = calibmat[self.num[:7]]['R']
        self.T_list = calibmat[self.num[:7]]['T']
        self.E_list = calibmat[self.num[:7]]['E']
        self.F_list = calibmat[self.num[:7]]['F']

    def Print_mat(self):    
        print('结果:\n')
        M1=np.array(self.mtx_1)
        print('M1:\n',M1)
        d1=np.array(self.dist_1)
        print('d1:\n',d1)
        M2=np.array(self.mtx_2)
        print('M2:\n',M2)
        d2=np.array(self.dist_2)
        print('d2:\n',d2)
        R=np.array(self.R_list)
        print('R:\n',R)
        T=np.array(self.T_list)
        print('T:\n',T)
        # EF为本征矩阵和基本矩阵，本功能暂时未使用
        print("EF为本征矩阵和基本矩阵，本功能暂时未使用")
        E=np.array(self.E_list)
        print('E:\n',E)
        F=np.array(self.F_list)
        print('F:\n',F)

    def stereoReMap(self, cal_path):
        M1=np.array(self.mtx_1)
        d1=np.array(self.dist_1)
        M2=np.array(self.mtx_2)
        d2=np.array(self.dist_2)
        R=np.array(self.R_list)
        T=np.array(self.T_list)
        # 左相机内参
        left_camera_matrix = M1
        left_distortion = d1
        # 右相机内参
        right_camera_matrix = M2
        right_distortion = d2
        # 平移关系向量
        T = T
        # 旋转关系向量
        # om = np.array([0.01911, 0.03125, -0.00960])
        R = R
        # 使用Rodrigues公式变换将旋转向量om变换为旋转矩阵R （也可以逆操作）
        # 也可以直接输用旋转矩阵
        size = (5472, 3648)  # 图像尺寸
        images_left = glob.glob(cal_path + 'Image0' + '*.bmp')
        images_right = glob.glob(cal_path + 'Image2' + '*.bmp')
        images_left.sort()
        images_right.sort()
        
        flags = 0
        flags |= cv2.CALIB_ZERO_DISPARITY
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix,
                                                                        left_distortion,
                                                                        right_camera_matrix,
                                                                        right_distortion,
                                                                        size, R, T,flags, alpha=-1)
        # 计算更正map

        # img_disp = np.zeros((5472, 3648),dtype=np.int32)
        # img_disp[1727,2482] = 407
        # img_deep = cv2.reprojectImageTo3D(img_disp, Q)

        left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,
                                                        left_distortion,
                                                        R1, P1, size,
                                                        cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,
                                                                right_distortion,
                                                                R2, P2, size,
                                                                cv2.CV_16SC2)
        for i, fname in enumerate(images_right):
            print(r'第',i+1,'次循环')
            imgL = cv2.imread(images_left[i])
            imgR = cv2.imread(images_right[i])
        # 进行立体更正
            imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
            imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
            # cv2.imshow("LeftPic", imgL)
            # cv2.imshow("LeftPicRec",imgL_rectified)
            # cv2.imshow("RightPic", imgR)
            # cv2.imshow("RightPicRec", imgR_rectified)
            concat_img = np.concatenate((imgL_rectified, imgR_rectified), axis=1)
        # 在纵向每 100 像素处绘制一条横向的绿色辅助线
            for y in range(0, concat_img.shape[0], 100):
                cv2.line(concat_img, (0, y), (concat_img.shape[1], y), (0, 255, 0), thickness=1)

            if os.path.exists(self.cal_path + 'rec_img'):
                pass
            else:
                os.mkdir(self.cal_path + 'rec_img')
                os.mkdir(self.cal_path + 'concat_img')
            doubleRectifyL = "Image0-"
            doubleRectifyR = "Image2-"
            # cv2.imwrite(self.cal_path + 'rec_img/' + 'LeftPicRec'+ ('{:04d}'.format(i)) + '.bmp',imgL_rectified)
            # cv2.imwrite(self.cal_path + 'rec_img/' + 'RightPicRec'+ ('{:04d}'.format(i)) + '.bmp',imgR_rectified)
            # cv2.imwrite(self.cal_path + 'concat_img/' + 'concat_img' + ('{:04d}'.format(i)) + '.bmp',concat_img)

            cv2.imwrite(self.cal_path + 'rec_img/' + doubleRectifyL + ('{:04d}'.format(i)) + '.bmp', imgL_rectified)
            cv2.imwrite(self.cal_path + 'rec_img/' + doubleRectifyR + ('{:04d}'.format(i)) + '.bmp', imgR_rectified)
            cv2.imwrite(self.cal_path + 'concat_img/' + 'concat_img' + ('{:04d}'.format(i)) + '.bmp', concat_img)
            cv2.waitKey(0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()    

    parser.add_argument('--filepath', default=r'./calculate/', help='String Filepath')
    base_num = 'base_06'
    args = parser.parse_args()
    # cal_data = StereoRectify(base_num, r'./calibrate/base_06/prefix_out/')
    cal_data = StereoRectify(base_num, args.filepath + 'base_06/01-142752/')
    # cal_data = StereoRectify(base_num, r'E:/stereo fire/23-04-11/base05/13541/')
    # cal_data = StereoRectify(base_num, args.filepath + 'base_01/01-1458/')
    # cal_data = StereoRectify(args.filepath + 'base_03/02-4952/')
    # cal_data = StereoRectify(args.filepath + 'base_03/03-8702/')
    # cal_data = StereoRectify(args.filepath + 'base_03/04-13831/')
    # cal_data = StereoRectify(args.filepath + 'base_04/01-13831/')
    # cal_data = StereoRectify(args.filepath + 'base_04/02-8710/')
    # cal_data = StereoRectify(args.filepath + 'base_04/03-5257/')
    # cal_data = StereoRectify(args.filepath + 'base_04/04-3721/')

