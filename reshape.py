import cv2
import glob
# 读取原始图片

def reshape_images(cal_path):

    images_left = glob.glob(cal_path +'Image0' + '*.bmp')
    images_right = glob.glob(cal_path + 'Image2' + '*.bmp')
    images_left.sort()
    images_right.sort()

    for i, fname in enumerate(images_right):
        print(r'第', i + 1, '次循环')
        num = str(i+23+4).zfill(4)
        img_l = cv2.imread(images_left[i])
        img_r = cv2.imread(images_right[i])
        # 获取原始图片的宽度和高度
        h, w = 3648, 5472
        # 设置目标图片的宽度和高度sceneflow
        # new_w = 960
        # new_h = 540
        #kitti2015
        new_w = 1242
        new_h = 375
        # 计算缩放比例
        scale_w = new_w / w
        scale_h = new_h / h

        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # img_l = clahe.apply(img_l)
        # img_r = clahe.apply(img_r)
        # img_l = cv2.convertScaleAbs(img_l, alpha=2, beta=5)
        # img_r = cv2.convertScaleAbs(img_r, alpha=2, beta=5)
        # 使用cv2.resize函数进行缩放，指定插值方法为cv2.INTER_AREA

        # resized_img_l = cv2.resize(img_l, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_AREA)
        # resized_img_r = cv2.resize(img_r, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_AREA)
        # cv2.imwrite(cal_path + '/Left' + num + '.bmp', img_l)
        # cv2.imwrite(cal_path + '/Right' + num + '.bmp', img_r)
        resized_img_l = cv2.resize(img_l, (1242,375), interpolation=cv2.INTER_AREA)
        resized_img_r = cv2.resize(img_r, (1242,375), interpolation=cv2.INTER_AREA)
        # x, y = 2000, 2400  # 起始点坐标
        #width, height = new_w, new_h  # 要裁剪的宽度和高度
        # resized_img_l = resized_img_l[y:y + height, x:x + width]
        # resized_img_r = resized_img_r[y:y + height, x:x + width]
        # resized_img_l = img_l[y:y + height, x:x + width]
        # resized_img_r = img_r[y:y + height, x-500:x - 500 + width]
        # resized_img_l = cv2.cvtColor(resized_img_l, cv2.COLOR_BGR2GRAY)
        # resized_img_r = cv2.cvtColor(resized_img_r, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(cal_path +'/image_2' + '/00' + num +'_10' + '.png', resized_img_l)
        cv2.imwrite(cal_path +'/image_3' + '/00' + num +'_10' + '.png', resized_img_r)

if __name__=='__main__':
    reshape_images('./photo/')