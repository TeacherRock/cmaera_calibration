import numpy as np
import cv2
import matplotlib.pyplot as plt


pic_num = 16     # 輸入圖片數
pic_size = (1280, 1024)   # 輸入圖片尺寸
board_w = 9   # 輸入圖片寬
board_h = 6   # 輸入圖片高

# 影像處理課的數據
# pic_size = (2048, 2048)   # 輸入圖片尺寸
# pic_num = 15     # 輸入圖片數
# board_w = 11   # 輸入圖片寬
# board_h = 8    # 輸入圖片高
# folder = 'Q2_Image/'
# data_tytpe = '.bmp'


# CALIB_CB_ADAPTIVE_THRESH : Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness).
# CALIB_CB_NORMALIZE_IMAGE : Normalize the image gamma with equalizeHist before applying fixed or adaptive thresholding.
# CALIB_CB_FAST_CHECK : Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed.
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

# 輸入圖片位置、圖片格式、存入圖片位置
folder = 'Pic/'
data_tytpe = '.jpg'
undis_folder = 'undis' + folder

# 創建棋盤格座標 ??  # 準備對象點
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)  # np.mgrid()返回多維結構 

# 創建等等用到的儲存陣列(拿來比較)
Pic_ini = cv2.imread(folder + '1' + data_tytpe)
Pic = [Pic_ini]*pic_num
Pic_dis = [Pic_ini]*pic_num
Pic_undis = [Pic_ini]*pic_num
Pic_gray = [Pic_ini]*pic_num
objpoints = [] 
imgpoints = [] 

# 設置終止條件，迭代30次或移動0.001??
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 讀取圖片
for i in range(pic_num):
    s = folder + str(i+1) + data_tytpe
    Pic[i] = cv2.imread(s)
    Pic_dis[i] = cv2.imread(s)
    Pic_undis[i] = cv2.imread(s)
    Pic_gray[i] =  cv2.cvtColor(Pic[i], cv2.COLOR_BGR2GRAY)  # 轉成灰階
    ret, corners = cv2.findChessboardCorners(Pic_gray[i], (board_w, board_h),None) # 找到棋盤角點
    corners = cv2.cornerSubPix(Pic_gray[i],corners, (11,11), (-1,-1), criteria)    # 更精確的點
    Pic[i] = cv2.drawChessboardCorners(Pic[i], (board_w, board_h), corners, ret)   # 將點畫上去
    objpoints.append(objp)    # 標定3維點(世界坐標系)
    imgpoints.append(corners) # 圖片上的2維點(影像坐標系)

# #IntrinsicMatrix and Distortion (內部參數cameraMatrix 與 失真矩陣dist)
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, Pic_gray[i].shape[::-1], None, None)
zero_dist = 0*dist

#Undistortion (取得去失真矩陣) 
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, pic_size, 0, pic_size)

for i in range(pic_num): 
    #Undistortion (去失真)
    Pic_undis[i] = cv2.undistort(Pic_undis[i], cameraMatrix, dist, None, newCameraMatrix)

    #ExtrinsicMatrix (取得外部參數,一張圖會有一組)
    _, R_exp, t = cv2.solvePnP(objpoints[i],np.ascontiguousarray(imgpoints[i][:,:2]).reshape((-1,1,2)),cameraMatrix,dist)
    R, _ = cv2.Rodrigues(R_exp)
    ExtrinsicMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for j in range(3):
        ExtrinsicMatrix[j][0] = R[j][0]
        ExtrinsicMatrix[j][1] = R[j][1]
        ExtrinsicMatrix[j][2] = R[j][2]
        ExtrinsicMatrix[j][3] = tvecs[0][j][0]
        ExtrinsicMatrix = np.array(ExtrinsicMatrix)

    # 畫線
    axis = np.float32([[2,2,-2], [2,0,-2], [0,0,-2], [0,2,-2], [2,2,0], [2,0,0], [0,0,0], [0,2,0]])
    imgpts, jac = cv2.projectPoints(axis, R, t, newCameraMatrix, zero_dist)
    imgpts = imgpts.astype(int)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[2].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[5].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[5].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[7].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)
    Pic_undis[i] = cv2.line( Pic_undis[i], tuple(imgpts[7].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)


cv2.namedWindow("Pic",0)
for i in range(pic_num):
    cv2.imshow('Pic',Pic_undis[i])
    cv2.waitKey(0)