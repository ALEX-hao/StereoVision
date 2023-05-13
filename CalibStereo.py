import numpy as np
import cv2
import os
import glob
import argparse
import time
import json
import tkinter as tk
from tkinter import filedialog


def browse_folder():
    global folder_path
    folder_path = filedialog.askdirectory()
    print("Selected folder:", folder_path)
    root.destroy()


class StereoCalibration(object):
    def __init__(self, basenum, filepath, gridDot):
        # termination
        # criteriaself.criteria = (cv2.TERM_CRITERIA_EPS +
        #                          cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
        #                              cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        # 迭代终止条件
        self.gridX = gridDot[0]
        self.gridY = gridDot[1]

        self.rt_left = float(0)
        self.rt_right = float(0)
        self.num = basenum

        # self.base_num=basenum
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.gridX * self.gridY, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.gridX, 0:self.gridY].T.reshape(-1, 2)
        self.objp = self.objp * 45.0

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.M1 = np.array([])
        self.d1 = np.array([])
        self.M2 = np.array([])
        self.d2 = np.array([])
        self.rt_left = 0.0
        self.rt_right = 0.0

        self.cal_path = filepath
        self.read_calibMat(self.cal_path)
        self.stereo_cal(self.cal_path)

    def read_calibMat(self, cal_path):
        # 读取json文件
        with open('calibCam' + '_' + self.num[:7] + '.json', 'r') as f:
            calibmat = json.load(f)
        self.M1 = np.array(calibmat[self.num[:7]]['mtx_1'])
        self.d1 = np.array(calibmat[self.num[:7]]['dist_1'])
        self.M2 = np.array(calibmat[self.num[:7]]['mtx_2'])
        self.d2 = np.array(calibmat[self.num[:7]]['dist_2'])
        self.rt_left = calibmat[self.num[:7]]['rt_left']
        self.rt_right = calibmat[self.num[:7]]['rt_right']

    def stereo_cal(self, cal_path):
        images_left = glob.glob(cal_path + '/Image0' + '*.bmp')
        images_right = glob.glob(cal_path + '/Image2' + '*.bmp')
        images_left.sort()
        images_right.sort()

        img0 = cv2.imread(self.cal_path + "/01.bmp")
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img_shape = gray0.shape[::-1]
        dims = img_shape

        for i, fname in enumerate(images_right):
            print(r'第', i + 1, '次循环')

            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            start_time = time.time()
            print('cv2.findChessboardCorners(L) is calling', start_time)
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (self.gridX, self.gridY), None)
            print('cv2.findChessboardCorners(R) is calling')
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (self.gridX, self.gridY), None)
            print("ret_l?", bool(ret_l))
            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            print(r'finish cost：', time.time() - start_time, 'second')

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                # self.imgpoints_l.append(corners_l)
                self.imgpoints_l.append(rt)
                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (self.gridX, self.gridY),
                                                  corners_l, ret_l)
                # cv2.imshow(images_left[i], img_l)
            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                ret_r = cv2.drawChessboardCorners(img_r, (self.gridX, self.gridY),
                                                  corners_r, ret_r)
                # self.imgpoints_r.append(corners_r)
                self.imgpoints_r.append(rt)

        # self.camera_model = self.stereo_calibrate(img_shape)


        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        print("stereocalib_criteria")
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 50, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,self.d2,
            dims, criteria=stereocalib_criteria, flags=flags)
        #   np.set_printoptions(suppress=True)
        #   不使用科学计数法
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        calibmat = {
            self.num[:7]: {
                'mtx_1': M1.tolist(),
                'dist_1': d1.tolist(),
                'mtx_2': M2.tolist(),
                'dist_2': d2.tolist(),
                'R': R.tolist(),
                'T': T.tolist(),
                'E': E.tolist(),
                'F': F.tolist(),
                'ret': ret,
                'rt_left': self.rt_left,
                'rt_right': self.rt_right
            }
        }
        with open(self.cal_path + "/calibMat_onlyStereo" + '_' + self.num[:7] + ".json", "w", encoding='utf-8') as f:
            json.dump(calibmat, f, ensure_ascii=False, indent=4)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('\n')


        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base_num ='base_03/'
    # parser.add_argument('--filepath', default=r'./calibrate/' + base_num, help='String Filepath')
    # parser.add_argument('--filepath', default=r'./calibrate/test/', help='String Filepath')
    # args = parser.parse_args()
    # cal_data = StereoCalibration(args.filepath, base_num)
    root = tk.Tk()
    root.title("Select Calibrate Folder")
    window_width = 400
    window_height = 100
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x, y))
    # 创建一个按钮，当点击它时，调用browse_folder函数
    browse_button = tk.Button(root,
                              text="————calibrate running————" + '\n' + "请点击此处选择标定图片路径",
                              command=browse_folder)
    browse_button.pack(pady=10)

    root.mainloop()

    path = folder_path
    print(path)
    print('\n')

    parser.add_argument('--filepath', default=path, help='String Filepath')
    parser.add_argument('--gridDot', default=(11, 8), help='gridDot as (11, 8)')
    parser.add_argument('--base_num', default='06', help='base_num as 00 01 02 …… 99')

    # parser.add_argument('--filepath', default=r'./calibrate/' + base_num, help='String Filepath')
    # parser.add_argument('--filepath', default=r'./calibrate/test/', help='String Filepath')
    args = parser.parse_args()
    base_num = 'base_' + args.base_num + r'/'
    cal_data = StereoCalibration(base_num, args.filepath, args.gridDot)
    os.system("pause")
