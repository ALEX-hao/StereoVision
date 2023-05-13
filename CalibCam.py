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


class CalibrationCam(object):
    def __init__(self, basenum, filepath, gridDot, sidename):
        # termination
        # criteriaself.criteria = (cv2.TERM_CRITERIA_EPS +
        #                          cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
        #                              cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        # 迭代终止条件
        self.sidename = sidename
        self.gridX = gridDot[0]
        self.gridY = gridDot[1]
        self.M1 = np.array([])
        self.d1 = np.array([])
        self.rt = float(0)
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
        self.imgpoints = []  # 2d points in image plane.

        self.cal_path = filepath
        self.cali_cam(self.cal_path)


    def cali_cam(self, cal_path):
        images = glob.glob(cal_path + '/' + self.sidename + '*.bmp')
        images.sort()

        img0 = cv2.imread(self.cal_path + "/01.bmp")
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img_shape = gray0.shape[::-1]

        for i, fname in enumerate(images):
            print(r'images角点识别第', i + 1, '次循环')
            img = cv2.imread(images[i])
            gray_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            start_time = time.time()
            print('cv2.findChessboardCorners is calling', start_time)
            ret, corners = cv2.findChessboardCorners(gray_l, (self.gridX, self.gridY), None)
            self.objpoints.append(self.objp)
            print(r'find corners',ret)
            print(r'finish cost：', time.time() - start_time, 'second')

            if ret is True:
                rt = cv2.cornerSubPix(gray_l, corners, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints.append(rt)
                # Draw and display the corners
                ret = cv2.drawChessboardCorners(img, (self.gridX, self.gridY),
                                                  corners, ret)
                img_s = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                cv2.imshow("img corner", img_s)
                cv2.waitKey(5)
            else:
                print()
                os.system("pause")

        print("cv2.calibrateCamera start")
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_shape, None, None)
        print("r1:", self.r1)
        print(cv2.Rodrigues(self.r1[0]))

        print("t1:", self.t1)
        self.rt = rt
        cv2.destroyAllWindows()
        # calibCam = {
        #     self.num[:7]: {
        #         'mtx_1': self.M1.tolist(),
        #         'dist_1': self.d1.tolist(),
        #         'mtx_2': self.M2.tolist(),
        #         'dist_2': self.d2.tolist(),
        #         'rt_left': self.rt,
        #         'rt_right': self.rt_right
        #     }
        # }
        # with open("calibCam" + '_' + self.num[:7] + ".json", "w", encoding='utf-8') as f:
        #     json.dump(calibCam, f, ensure_ascii=False, indent=4)


    def pullout(self):
        return (self.M1, self.d1, self.rt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    cal_data_L = CalibrationCam(base_num, args.filepath, args.gridDot, 'Image0')
    M1, d1, rt_left = cal_data_L.pullout()

    cal_data_R = CalibrationCam(base_num, args.filepath, args.gridDot, 'Image2')
    M2, d2, rt_right = cal_data_R.pullout()

    calibCam = {
        base_num[:7]: {
            'mtx_1': M1.tolist(),
            'dist_1': d1.tolist(),
            'mtx_2': M2.tolist(),
            'dist_2': d2.tolist(),
            'rt_left': rt_left,
            'rt_right': rt_right
        }
    }
    # with open("calibCam" + '_' + base_num[:7] + ".json", "w", encoding='utf-8') as f:
    #     json.dump(calibCam, f, ensure_ascii=False, indent=4)
    os.system("pause")
