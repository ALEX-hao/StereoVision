#   用于计算目标点的坐标
#   用于calibrMat的读取 输出 并对指定图片对矫正 输出矫正后图片 以及拼接图片
import json
import numpy as np
import os
import cv2
import argparse
import glob

class StereoCoordinate(object):
    def __init__(self, filepath , distance, base_num):
        self.cal_path = filepath
        print('call filepath = ' + self.cal_path)
        self.num = base_num
        self.gt=distance
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
        self.Coordinate(self.cal_path)

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

    def computexyz(self, pt_left, pt_right, M1, M2, R, T):
        # 补齐次项
        pt_left = np.pad(pt_left, pad_width=(0, 1), mode='constant', constant_values=1)
        pt_right = np.pad(pt_right, pad_width=(0, 1), mode='constant', constant_values=1)
        # 计算内参的逆乘以像素坐标
        pt_lCam = np.matmul(np.linalg.inv(M1), pt_left)  # 得到两个方程：xl = a*zl  ;  yl = b*zl
        pt_rCam = np.matmul(np.linalg.inv(M2), pt_right)  # 得到两个方程：xr = c*zr  ;  yr = d*zr
        # 将上面四个方程带入相机坐标转换公式，得到两个变量的三个方程
        pt_l2rCam = np.matmul(R, pt_lCam)  ##左边系数##【e,f,g】
        ##由前面两个方程得到z_l = (c*T[1]-d*T[0])/(e*d-f*c) , z_r = (e*z_l +T[0])/c
        z_l_1 = (pt_rCam[0] * T[1] - pt_rCam[1] * T[0]) / (
                    pt_l2rCam[0] * pt_rCam[1] - pt_l2rCam[1] * pt_rCam[0])
        z_r_1 = (pt_l2rCam[0] * z_l_1 + T[0]) / pt_rCam[0]
        ##由第一、第三两个方程得到z_l = (T[0]-c*T[2])/(g*c-e) , z_r = g*z_l +T[2]
        z_l_3 = (T[0] - pt_rCam[0] * T[2]) / (pt_l2rCam[2] * pt_rCam[0] - pt_l2rCam[0])
        z_r_3 = pt_l2rCam[2] * z_l_3 + T[2]
        z_l = (z_r_1 + z_l_3) / 2.0
        x_y_z = pt_lCam * z_l
        return tuple(x_y_z)
    # 该函数尚未测试
    def computeXYZ(self, pt_left, pt_right, M1, M2, R, T):
        # 补齐次项
        pt_left = np.hstack((pt_left, 1))
        pt_right = np.hstack((pt_right, 1))
        # 计算内参的逆乘以像素坐标
        tmp_l = np.linalg.inv(M1) @ pt_left  # 得到两个方程：xl = a*zl  ;  yl = b*zl
        tmp_r = np.linalg.inv(M2) @ pt_right  # 得到两个方程：xr = c*zr  ;  yr = d*zr
        # 将上面四个方程带入相机坐标转换公式，得到两个变量的三个方程
        left_coef = R @ tmp_l  ##左边系数##【e,f,g】
        right_coef = np.array([tmp_r[0], tmp_r[1], -1])  ##右边系数##【c,d,-1】
        # 用np.linalg.solve求解线性方程组
        z_l_z_r = np.linalg.solve(np.vstack((left_coef, right_coef)), T)
        z_l = z_l_z_r[0]
        x_y_z = tmp_l * z_l
        return tuple(x_y_z)

    def groundturth(self, pt_left, M1, Z ):
        point = np.pad(pt_left, pad_width=(0, 1), mode='constant', constant_values=1)
        tmp = np.matmul(np.linalg.inv(M1), point)
        z = Z ** 2 / np.sum((tmp) ** 2)
        z = np.sqrt(z)
        x_y_z = tmp * z
        return x_y_z

    def leastsq_Compu(self,pt_left, pt_right, rtLeft, rtRight):
        u_left = pt_left[0]
        v_left = pt_left[1]
        u_right = pt_right[0]
        v_right = pt_right[1]
        A = np.zeros(shape=(4, 3))
        for i in range(0, 3):
            A[0][i] = u_left * rtLeft[2, i] - rtLeft[0][i]
        for i in range(0, 3):
            A[1][i] = v_left * rtLeft[2][i] - rtLeft[1][i]
        for i in range(0, 3):
            A[2][i] = u_right * rtRight[2][i] - rtRight[0][i]
        for i in range(0, 3):
            A[3][i] = v_right * rtRight[2][i] - rtRight[1][i]
        B = np.zeros(shape=(4, 1))
        for i in range(0, 2):
            B[i][0] = rtLeft[i][3] - u_left * rtLeft[2][3]
        for i in range(2, 4):
            B[i][0] = rtRight[i - 2][3] - u_right * rtRight[2][3]

        XYZ = np.zeros(shape=(3, 1))
        ret, XYZ = cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)
        return XYZ

    def traingularpt(self, pt_left, pt_right, M1, M2, R, T ):
        R_left = np.eye(3)
        T_left = np.array([0, 0, 0])
        T_left = T_left.reshape(3, 1)

        # 对输入像素坐标进行归一化
        # pt_left_und = np.linalg.inv(M1) @ np.array([pt_left[0],pt_left[1],1])
        # pt_right_und = np.linalg.inv(M2) @ np.array([pt_right[0],pt_right[1],1])

        pt_left_und = np.array([pt_left[0], pt_left[1],1])
        pt_right_und = np.array([pt_right[0], pt_right[1], 1])
        src_ptleft = np.array([[pt_left_und[0], pt_left_und[1]]],
                              dtype=np.float32)
        src_ptright = np.array([[pt_right_und[0], pt_right_und[1]]],
                              dtype=np.float32)
        # 计算输入像素坐标的undistort
        left_undistorted = cv2.undistortPoints(src_ptleft, np.array(M1),
                                               np.array(self.dist_1))
        right_undistorted = cv2.undistortPoints(src_ptright, np.array(M2),
                                                np.array(self.dist_2))
        # 对左右相机的旋转平移矩阵进行归一化
        rtLeft = M1 @ np.hstack([R_left, T_left])
        rtRight = M2 @ np.hstack([R, T])



        #
        pt_4d2 = cv2.triangulatePoints(rtLeft,
                                      rtRight,
                                      left_undistorted.transpose(),
                                      right_undistorted.transpose())
        print("pt_4d2: \n", pt_4d2)

        pt_3d = pt_4d2[:3]/pt_4d2[3]
        pt_3d = pt_3d.reshape(-1, 1)
        return pt_3d

    def Coordinate(self, cal_path):
        M1=np.array(self.mtx_1)        
        d1=np.array(self.dist_1)        
        M2=np.array(self.mtx_2)        
        d2=np.array(self.dist_2)        
        R=np.array(self.R_list)
        T=np.array(self.T_list)
        # 左相机内参
        # left_camera_matrix = M1
        # left_distortion = d1
        # 右相机内参
        # right_camera_matrix = M2
        # right_distortion = d2     
        size = (5472, 3648)  # 图像尺寸
        images_left = glob.glob(cal_path + 'Left' + '*.bmp')
        images_right = glob.glob(cal_path + 'Right' + '*.bmp')
        images_left.sort()
        images_right.sort()
        errors = []

        for i, fname in enumerate(images_right):
            print(r'第',i+1,'次循环')
            save_num = i
            imgL = cv2.imread(images_left[i])
            imgR = cv2.imread(images_right[i])
            u_left,v_left=[0,0]
            u_right,v_right=[0,0]
            blue1=(200,0,0)
            blue2=(255,0,0)
            # 定义颜色阈值范围
            # 比较图像中的颜色与颜色阈值范围
            maskL = cv2.inRange(imgL,blue1,blue2)
            # 查找mask中第一个非零值的索引
            y, x = np.where(maskL != 0)
            if len(x) > 0:               
                # print("Left@x:", x[0], "y:", y[0])
                u_left=x[0]
                v_left=y[0]
            else:
                print("No matching pixel found in L.")
            maskR = cv2.inRange(imgR,blue1,blue2)
            # 查找mask中第一个非零值的索引
            y, x = np.where(maskR != 0)
            if len(x) > 0:               
                # print("Right@x:", x[0], "y:", y[0])
                u_right=x[0]
                v_right=y[0]
            else:
                print("No matching pixel found in R.")

            pt_left = np.array([u_left, v_left])
            pt_right = np.array([u_right, v_right])


            # ground truth计算
            # x_y_z = self.groundturth(pt_left, M1, self.gt)
            x_y_z = self.groundturth(pt_right, M2, self.gt)
            R_left = np.eye(3)
            T_left = np.array([0, 0, 0])
            T_left = T_left.reshape(3, 1)
            rtRight = M2 @ np.hstack([R, T])
            rtLeft = M1 @ np.hstack([R_left, T_left])
            XYZ = self.leastsq_Compu(pt_left, pt_right, rtLeft, rtRight)
            # XYZ = self.computexyz(pt_left, pt_right, M1, M2, R, T)
            # XYZ = self.traingularpt(pt_left, pt_right, M1, M2, R, T)

            text = "("+str(XYZ[0]) + " , " + str(XYZ[1]) + " , " + str(XYZ[2]) + ")"
            textgt = "("+str(x_y_z[0]) + " , " + str(x_y_z[1]) + " , " + str(x_y_z[2]) + ")"
            error_distance = np.sqrt((x_y_z[0] - XYZ[0]) ** 2 + (x_y_z[1] - XYZ[1]) ** 2 + (x_y_z[2] - XYZ[2]) ** 2)
            print('the %03d points-pair ErrorDistance  =  %.4f mm' % (i, error_distance))
            errors.append(error_distance)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            color2 = (0, 255, 0)
            thickness = 3
            img_save = cv2.putText(imgL, text, (u_left+1000+save_num, v_left-100-save_num),
                                   font, fontScale, color, thickness, cv2.LINE_AA)
            img_save = cv2.putText(imgL, textgt, (u_left+1000+save_num, v_left-50-save_num),
                                   font, fontScale, color2, thickness, cv2.LINE_AA)
            if os.path.exists(self.cal_path + 'coordinate_img'):
                pass
            else:
                os.mkdir(self.cal_path + 'coordinate_img')
            filename = self.cal_path + 'coordinate_img/' + 'Pic'+ str('{:02d}'.format(save_num))

            print(text)
            print(textgt)
            cv2.imwrite(filename + '.bmp', img_save)
            cv2.waitKey(5)
        errors = np.mean(np.array(errors))
        print('the averaged points-pair ErrorDistance  =  %.4f mm' % errors)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    base_num = 'base_06/'

    parser.add_argument('--filepath', default=r'./calculate/'+ base_num, help='String Filepath')
    args = parser.parse_args()

    # base_06
    cal_data = StereoCoordinate(args.filepath + '01-142752/rec_img/point_handcraft/', 142752, base_num)
    # base_05

    # base_04
    # cal_data = StereoCoordinate(args.filepath + '01-13831/rec_img/', 138310, base_num)
    # cal_data = StereoCoordinate(args.filepath + '02-8710/rec_img/', 87100, base_num)
    # cal_data = StereoCoordinate(args.filepath + '03-5257/rec_img/', 52570, base_num)
    # cal_data = StereoCoordinate(args.filepath + '04-3721/rec_img/', 37210, base_num)

    # base_03
    # cal_data = StereoCoordinate(args.filepath + '01-2322/rec_img/', 23220, base_num)
    # cal_data = StereoCoordinate(args.filepath + '02-4952/rec_img/', 49520, base_num)
    # cal_data = StereoCoordinate(args.filepath + '03-8702/rec_img/', 87020, base_num)
    # cal_data = StereoCoordinate(args.filepath + '04-13831/rec_img/', 138310, base_num)

    # base_01
    # cal_data = StereoCoordinate(args.filepath + '01-1458/point handcraft/',14580, base_num)
    # cal_data = StereoCoordinate(args.filepath + '02-3283/point handcraft/',32830, base_num)
    # cal_data = StereoCoordinate(args.filepath + '03-6056/point handcraft/',60560, base_num)
    # cal_data = StereoCoordinate(args.filepath + '04-9827/point handcraft/',98270, base_num)
    # cal_data = StereoCoordinate(args.filepath + '05-13976/point handcraft/',139760, base_num)
