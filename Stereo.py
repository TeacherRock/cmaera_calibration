import numpy as np
import cv2
import matplotlib.pyplot as plt

## cd D:\成大\碩一\新訓\我的\5_相機校正

pic_num = 10     # 輸入圖片數
pic_size = (1280, 1024)   # 輸入圖片尺寸
board_w = 9   # 輸入圖片寬
board_h = 6   # 輸入圖片高
data_tytpe = '.bmp'  # 圖片格式

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


for eye in range(2):
    if eye == 0:
       # 輸入圖片位置、存入圖片位置
       folder = 'Pic_R/'
       undis_folder = 'undis/' + folder 
    elif eye == 1: 
       # 輸入圖片位置、存入圖片位置
       folder = 'Pic_L/'
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
        cv2.namedWindow('image')
        cv2.imshow('image', Pic[i])
        cv2.waitKey()
        cv2.destroyAllWindows()
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

    #Undistortion (取得去失真矩陣) 第4個參數會為0會把無效影像刪掉，1則保留
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, pic_size, 0, pic_size)

    for i in range(pic_num): 
        #Undistortion (去失真)
        Pic_undis[i] = cv2.undistort(Pic_undis[i], cameraMatrix, dist, None, newCameraMatrix)
        Pic_undis_line[i] = cv2.undistort(Pic_undis[i], cameraMatrix, dist, None, newCameraMatrix)

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
        imgpts, jac = cv2.projectPoints(axis, R, t, cameraMatrix, zero_dist)
        imgpts = imgpts.astype(int)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[2].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[5].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[5].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[7].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)
        Pic_undis_line[i] = cv2.line( Pic_undis_line[i], tuple(imgpts[7].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
        s = undis_folder + str(i+1) + data_tytpe
        # cv2.imwrite(s, Pic_undis[i])
    cv2.namedWindow('image')
    for i in range(10):
        cv2.imshow('image', Pic_undis[i])
        cv2.waitKey()
    cv2.destroyAllWindows()




       
# RedPointL = cv2.imread('Pic_RedPoint/RedPointL.jpg', 0)
# RedPointR = cv2.imread('Pic_RedPoint/RedPointR.jpg', 0)
# # RedPointL = cv2.imread('Pic_Stereo/Stereo1-L.bmp', 0)
# # RedPointR= cv2.imread('Pic_Stereo/Stereo1-R.bmp', 0)

# # 求disaprity
# # cv2.StereoBM_create(numDisparities, blockSize) : 建構深度圖 
# # numDisparities : 最大視差值與最小視差值之差(必須是16的整數倍)
# # blockSize : 匹配的塊的大小，必須為>=1的奇數，通常在3~11之間
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=11)
# disparity = stereo.compute(RedPointL, RedPointR)  # 計算 disparity
# final_img = cv2.normalize(disparity,  disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# resimg = cv2.resize(final_img, (640,480), interpolation=cv2.INTER_CUBIC)

# text="disparity"
# Baseline = 75 # mm
# f = 6*1920/7.11 # 未知

# # 滑鼠按下的動作
# def OnMouseAction(event,x,y,flags,parm):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         b = 254 + resimg[y][x]
#         print(resimg[y][x],"pixel")
#         z = Baseline*f/b # 計算深度
#         z = int(z)
#         print(z,"mm")
#         dis = "disparity: {:d}pixel".format(b)
#         dep = "Depth: {:d}mm".format(z)
#         cv2.rectangle(resimg, (640, 480), (400,400), (255, 255, 255), thickness=-1)
#         cv2.putText(resimg, dis, (420,420), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0, 0),1, cv2.LINE_AA)
#         cv2.putText(resimg, dep, (420,460), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0, 0),1, cv2.LINE_AA)

# # 顯示        
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', OnMouseAction)  
# while(1):
#     cv2.imshow('image', resimg)
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break
# cv2.destroyAllWindows()
