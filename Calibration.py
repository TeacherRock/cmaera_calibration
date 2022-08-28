import numpy as np
import cv2
import matplotlib.pyplot as plt


pic_num = 20     # 輸入圖片數
pic_size = (1024, 576)   # 輸入圖片尺寸
board_w = 11   # 輸入圖片寬
board_h = 7   # 輸入圖片高
data_tytpe = '.png'  # 圖片格式


# CALIB_CB_ADAPTIVE_THRESH : Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness).
# CALIB_CB_NORMALIZE_IMAGE : Normalize the image gamma with equalizeHist before applying fixed or adaptive thresholding.
# CALIB_CB_FAST_CHECK : Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed.
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

for eye in range(2):
    if eye == 0:
       # 輸入圖片位置、存入圖片位置
       folder = 'chessboard_data/imgs/rightcamera/Im_R_'
       undis_folder = 'undis/' + folder 
    elif eye == 1: 
       # 輸入圖片位置、存入圖片位置
       folder = 'chessboard_data/imgs/leftcamera/Im_L_'
       undis_folder = 'undis/' + folder 
    # 創建棋盤格座標 # 準備對象點
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)  # np.mgrid()返回多維結構 

    # 創建等等用到的儲存陣列(拿來比較)
    Pic_ini = cv2.imread(folder + '1' + data_tytpe)
    Pic = [Pic_ini]*pic_num
    Pic_undis = [Pic_ini]*pic_num
    Pic_undis_line = [Pic_ini]*pic_num
    Pic_gray = [Pic_ini]*pic_num
    objpoints = [] 
    imgpoints = [] 

    # 設置終止條件，迭代30次或移動0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 讀取圖片
    for i in range(pic_num):
        s = folder + str(i+1) + data_tytpe
        Pic[i] = cv2.imread(s)
        Pic_undis[i] = cv2.imread(s)
        Pic_undis_line[i] = cv2.imread(s)
        Pic_gray[i] =  cv2.cvtColor(Pic[i], cv2.COLOR_BGR2GRAY)  # 轉成灰階
        ret, corners = cv2.findChessboardCorners(Pic_gray[i], (board_w, board_h),None) # 找到棋盤角點
        corners = cv2.cornerSubPix(Pic_gray[i],corners, (11,11), (-1,-1), criteria)    # 更精確的點
        Pic[i] = cv2.drawChessboardCorners(Pic[i], (board_w, board_h), corners, ret)   # 將點畫上去
        objpoints.append(objp)    # 標定3維點(世界坐標系)
        imgpoints.append(corners) # 圖片上的2維點(影像坐標系)

    # #IntrinsicMatrix and Distortion (內部參數cameraMatrix 與 失真矩陣dist)
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, Pic_gray[i].shape[::-1], None, None)
    zero_dist = 0*dist

    #Undistortion 第4個參數會為0會把無效影像刪掉，1則保留
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, pic_size, 0, pic_size)

    if eye == 0:
        cameraMatrixR = cameraMatrix
        distR = dist
        imgpointsR = imgpoints
    else:
        cameraMatrixL = cameraMatrix
        distL = dist
        imgpointsL = imgpoints

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL, 
    distL, cameraMatrixR, distR, pic_size)
# print(cameraMatrixR, distR)    
# print(cameraMatrixL, distL)   
# print(R, T)

# data = np.load('chessboard_data/out/parameters.npz')
# print(data.files)