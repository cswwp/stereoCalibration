import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients
import os



def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8, #* 3,# * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32, #* 3,# * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    #left_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=11)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--left_src', type=str, required=True)
    parser.add_argument('--right_src', type=str, required=True)
    args = parser.parse_args()

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    left_content = []
    right_content = []

    left_content = sorted([os.path.join(args.left_src, sub) for sub in os.listdir(args.left_src)])
    right_content = sorted([os.path.join(args.right_src, sub) for sub in os.listdir(args.right_src)])

    line_num = 10
    line_x0 = [0 for i in range(480//line_num)]
    line_y = [i for i in range(0, 480, 480//line_num)]
    line_x1 = [640*2 - 1 for i in range(480//line_num)]

    start = tuple(zip(line_x0, line_y))
    end = tuple(zip(line_x1, line_y))

    for i in range(len(left_content)):
        leftFrame = cv2.imread(left_content[i])
        leftFrame = cv2.resize(leftFrame, (640, 480))
        rightFrame = cv2.imread(right_content[i])
        rightFrame = cv2.resize(rightFrame, (640, 480))

        height, width, channel = leftFrame.shape  # We will use the shape for remap

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        merge = np.hstack((left_rectified, right_rectified))



        for index in range(len(start)):
            s = start[index]
            e = end[index]
            show_remap = cv2.line(merge, s, e, (255, 0, 0), thickness=1)

        cv2.imshow('remap', show_remap)

        #cv2.

        cv2.waitKey(0)

