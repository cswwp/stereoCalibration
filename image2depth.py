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
        minDisparity=8,
        numDisparities=16*3,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 1 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 1 * window_size,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
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

    cv2.imshow('odisl', np.int8(displ))
    cv2.imshow('odisr', np.int8(dispr))

    filteredImgL = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImgL = cv2.normalize(src=filteredImgL, dst=filteredImgL, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImgL = np.uint8(filteredImgL)


    filteredImgR = wls_filter.filter(dispr, imgR, None, displ)  # important to put "imgL" here!!!

    filteredImgR = cv2.normalize(src=filteredImgR, dst=filteredImgR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImgR = np.uint8(filteredImgR)


    return filteredImgL, filteredImgR

def height_line(height, width):
    line_num = 10
    line_x0 = [0 for i in range(height // line_num)]
    line_y = [i for i in range(0, height, height // line_num)]
    line_x1 = [width * 2 - 1 for i in range(height // line_num)]

    start = tuple(zip(line_x0, line_y))
    end = tuple(zip(line_x1, line_y))

    return start, end


if __name__ == '__main__':

    base = 'image_transform'
    # base = '/Users/parker/Downloads/Desktop/photo/1black/cropimg_roi_result'
    # base = '/Users/parker/Downloads/Desktop/photo/2black/cropimg_roi_result'
    # base = '/Users/parker/Downloads/Desktop/photo/1color/cropimg_roi_result'
    #
    # base = '/Users/parker/Downloads/Desktop/photo/2color/cropimg_roi_result'
    # base = '/Users/parker/Downloads/Desktop/photo/1black/'
    # base = '/Users/parker/Downloads/Desktop/photo/2black/'
    # base = '/Users/parker/Downloads/Desktop/photo/2color/'
    # base = '/Users/parker/Downloads/Desktop/photo/1color/'

    prefix_left = 'image0'
    prefix_right = 'image1'

    l_path = os.path.join(base, prefix_left)

    # l_path = '/Users/parker/Downloads/Desktop/photo/left/1black'
    r_path = os.path.join(base, prefix_right)
    # r_path = '/Users/parker/Downloads/Desktop/person/right'
    #r_path = '/Users/parker/Downloads/Desktop/photo/right/1black'
    l_lst = sorted([os.path.join(l_path, fn) for fn in os.listdir(l_path)])
    r_lst = sorted([os.path.join(r_path, fn) for fn in os.listdir(r_path)])

    save_l_dis_path = os.path.join(base, 'disl')
    save_r_dis_path = os.path.join(base, 'disr')

    if not os.path.exists(save_l_dis_path):
        os.makedirs(save_l_dis_path)

    if not os.path.exists(save_r_dis_path):
        os.makedirs(save_r_dis_path)


    count = 100

    for num in range(len(l_lst)):


        print(l_lst[num], r_lst[num])
        left_rectified = cv2.imread(l_lst[num], 0)
        right_rectified = cv2.imread(r_lst[num], 0)
        assert left_rectified.shape == right_rectified.shape
        # left_rectified = cv2.resize(left_rectified, (width, height))
        # right_rectified = cv2.resize(right_rectified, (width, height))

        fn_l = l_lst[num].split('/')[-1]
        fn_r = r_lst[num].split('/')[-1]


        height, width = left_rectified.shape  # We will use the shape for remap

        merge = np.hstack((left_rectified, right_rectified))

        start, end = height_line(height, width)

        for index in range(len(start)):
            s = start[index]
            e = end[index]
            show_remap = cv2.line(merge, s, e, (255, 0, 0), thickness=1)
        cv2.imshow('merge', show_remap)

        # We need grayscale for disparity map.
        gray_left = left_rectified #cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = right_rectified #cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disparity_image_l, disparity_image_r = depth_map(gray_left, gray_right)  # Get the disparity map
        # Show the images

        cv2.imshow('Disparity-l', disparity_image_l)
        cv2.imshow('Disparity-r', disparity_image_r)

        # cv2.imwrite(os.path.join(save_l_dis_path, fn_l), disparity_image_l)
        # cv2.imwrite(os.path.join(save_r_dis_path, fn_r), disparity_image_r)

        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(os.path.join(l_save_rectify_path, str(count) + '.jpg'), left_rectified)
            cv2.imwrite(os.path.join(r_save_rectify_path, str(count) + '.jpg'), right_rectified)
            print('saved***8')
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break

    # Release the sources.
    # cap_left.release()
    # cap_right.release()
    cv2.destroyAllWindows()
