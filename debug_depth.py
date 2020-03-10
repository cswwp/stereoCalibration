import numpy as np
import cv2
import argparse
import sys
from calibration_store import load_stereo_coefficients
def update(val = 0):
    # left_matcher.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    # left_matcher.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    # left_matcher.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    # left_matcher.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    # left_matcher.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))
    global window_size
    global blockSize
    global uniquenessRatio
    global speckleRange
    global speckleWindowSize
    global disp12MaxDiff
    global num_disp

    num_disp = cv2.getTrackbarPos('num_disp', 'Disparity')*16
    window_size = cv2.getTrackbarPos('window_size', 'Disparity')
    blockSize = window_size
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Disparity')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'Disparity')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Disparity')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Disparity')


def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    # window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    # left_matcher = cv2.StereoSGBM_create(
    #     minDisparity=-1,
    #     numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    #     blockSize=window_size,
    #     P1=8 * 3 * window_size,
    #     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    #     P2=32 * 3 * window_size,
    #     disp12MaxDiff=12,
    #     uniquenessRatio=10,
    #     speckleWindowSize=50,
    #     speckleRange=32,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )

    # num_disp = 112 - min_disp.get()
    # left_matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size.get())
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
    parser.add_argument('--left_source', type=int, required=True, help='Left video or v4l2 device name')
    parser.add_argument('--right_source', type=int, required=True, help='Right video or v4l2 device name')
    parser.add_argument('--is_real_time', type=int, required=True, help='Is it camera stream or video')
    parser.add_argument('--highR', type=int,default=0)


    args = parser.parse_args()

    if args.highR:
        width = 960
        height = 540
    else:
        width = 640
        height = 480
    l_save_rectify_path = 'l_rectify'
    r_save_rectify_path = 'r_rectify'
    import os
    if not os.path.exists(l_save_rectify_path):
        os.makedirs(l_save_rectify_path)
    if not os.path.exists(r_save_rectify_path):
        os.makedirs(r_save_rectify_path)


    # is camera stream or video
    if args.is_real_time:
        cap_left = cv2.VideoCapture(args.left_source, cv2.CAP_V4L2)
        cap_right = cv2.VideoCapture(args.right_source, cv2.CAP_V4L2)
    else:
        cap_left = cv2.VideoCapture(args.left_source)
        cap_right = cv2.VideoCapture(args.right_source)

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    # print(K1)
    # print(D1)
    # print(R1) P1)
    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't opened the streams!")
        sys.exit(-9)

    # Change the resolution in need
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # float
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # float
    #
    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # float
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # float

    line_num = 10
    line_x0 = [0 for i in range(height // line_num)]
    line_y = [i for i in range(0, height, height // line_num)]
    line_x1 = [width * 2 - 1 for i in range(height // line_num)]


    start = tuple(zip(line_x0, line_y))
    end = tuple(zip(line_x1, line_y))

    count = 100

######################################
    window_size = 5
    min_disp = 16
    num_disp = 16
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1_match = 600
    P2_match = 2400

    cv2.namedWindow('Disparity')
    cv2.createTrackbar('speckleRange', 'Disparity', speckleRange, 50, update)
    cv2.createTrackbar('num_disp', 'Disparity', num_disp, 10, update)
    cv2.createTrackbar('window_size', 'Disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'Disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'Disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'Disparity', disp12MaxDiff, 250, update)


    #################################################################

    while True:  # Loop until 'q' pressed or stream ends
        # Grab&retreive for sync images
        if not (cap_left.grab() and cap_right.grab()):
            print("No more frames")
            break

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            uniquenessRatio=uniquenessRatio,
            speckleRange=speckleRange,
            speckleWindowSize=speckleWindowSize,
            disp12MaxDiff=disp12MaxDiff,
            P1=P1_match,
            P2=P2_match
        )

        _, leftFrame = cap_left.retrieve()
        leftFrame = cv2.resize(leftFrame, (width, height))

        #print('leftFrame:', leftFrame.shape)


        _, rightFrame = cap_right.retrieve()
        rightFrame = cv2.resize(rightFrame, (width, height))

        #print('rightFrame:', rightFrame.shape)

        height, width, channel = leftFrame.shape  # We will use the shape for remap

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        #print('right_rectified:', right_rectified.shape)

        # left_rectified = left_rectified[200:400, 300:700]
        # right_rectified = right_rectified[200:400, 300:700]

        merge = np.hstack((left_rectified, right_rectified))

        for index in range(len(start)):
            s = start[index]
            e = end[index]
            show_remap = cv2.line(merge, s, e, (255, 0, 0), thickness=1)

        cv2.imshow('remap', show_remap)
        #cv2.waitKey(0)

        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disparity_image = depth_map(gray_left, gray_right)  # Get the disparity map
        # Show the images
        cv2.imshow('left(R)', leftFrame)
        cv2.imshow('right(R)', rightFrame)
        cv2.imshow('Disparity', disparity_image)
        if cv2.waitKey(500) & 0xFF == ord('c'):
            cv2.imwrite(os.path.join(l_save_rectify_path, str(count) + '.jpg'), left_rectified)
            cv2.imwrite(os.path.join(r_save_rectify_path, str(count) + '.jpg'), right_rectified)
            print('saved***8')
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break

    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
