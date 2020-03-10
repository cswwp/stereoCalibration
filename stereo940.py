import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients
import os

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=32,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 10000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    cv2.imshow('odis', np.int8(displ))

    # disp_temp = cv2.normalize(displ, displ, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow('displ', disp_temp)


    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--im_h', type=int, default=1280)
    parser.add_argument('--im_w', type=int, default=720)

    args = parser.parse_args()

    base = args.test_dir#'image_transform'

    height = args.im_h
    width = args.im_w


    l_save_rectify_path = os.path.join(base, 'l_rectify')
    r_save_rectify_path = os.path.join(base, 'r_rectify')
    import os
    if not os.path.exists(l_save_rectify_path):
        os.makedirs(l_save_rectify_path)
    if not os.path.exists(r_save_rectify_path):
        os.makedirs(r_save_rectify_path)



    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    print('K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q:', K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)
    line_num = 10
    line_x0 = [0 for i in range(height // line_num)]
    line_y = [i for i in range(0, height, height // line_num)]
    line_x1 = [width * 2 - 1 for i in range(height // line_num)]


    start = tuple(zip(line_x0, line_y))
    end = tuple(zip(line_x1, line_y))

    left_src_dir = os.path.join(base, 'image0')
    right_src_dir = os.path.join(base, 'image1')

    left_lst = os.listdir(left_src_dir)
    right_lst = os.listdir(right_src_dir)

    left_lst = sorted(left_lst)
    right_lst = sorted(right_lst)

    for index_local in range(len(left_lst)):
        print('left:', os.path.join(left_src_dir, left_lst[index_local]))
        leftFrame = cv2.imread(os.path.join(left_src_dir, left_lst[index_local]), 0)
        leftFrame = cv2.resize(leftFrame, (width, height))

        print('right:', os.path.join(left_src_dir, left_lst[index_local]))
        rightFrame = cv2.imread(os.path.join(right_src_dir, right_lst[index_local]), 0)
        rightFrame = cv2.resize(rightFrame, (width, height))

        #height, width = leftFrame.shape  # We will use the shape for remap
        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)

        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


        merge = np.hstack((left_rectified, right_rectified))
        #print('fdsafsdafsdA:')
        for index in range(len(start)):
            s = start[index]
            e = end[index]
            show_remap = cv2.line(merge, s, e, (0), thickness=1)

        cv2.imshow('remap', cv2.resize(show_remap, (720, 640)))

        disparity_image = depth_map(left_rectified, right_rectified)  # Get the disparity map
        # Show the images
        cv2.imshow('left', cv2.resize(leftFrame, (360, 640)))
        cv2.imshow('right', cv2.resize(rightFrame, (360, 640)))
        cv2.imshow('Disparity', cv2.resize(disparity_image, (360, 640)))

        cv2.waitKey(0)


    cv2.destroyAllWindows()
