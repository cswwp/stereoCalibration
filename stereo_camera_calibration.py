import numpy as np
import cv2
import glob
import argparse
import sys
from calibration_store import load_coefficients, save_stereo_coefficients

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None
#image_size = (960, 540)


def stereo_calibrate(left_file, right_file, left_dir, left_prefix, right_dir, right_prefix, image_format, save_file, square_size, width=9, height=6):
    """ Stereo calibration and rectification """
    objp, leftp, rightp = load_image_points(left_dir, left_prefix, right_dir, right_prefix, image_format, square_size, width, height)

    K1, D1 = load_coefficients(left_file)
    K2, D2 = load_coefficients(right_file)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    #flags |= cv2.CALIB_SAME_FOCAL_LENGTH

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size, flags=flags)
    print("Stereo calibration rms: ", ret)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.6)

    save_stereo_coefficients(save_file, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)


def load_image_points(left_dir, left_prefix, right_dir, right_prefix, image_format, square_size, width=9, height=6):
    global image_size
    pattern_size = (width, height)  # Chessboard size!
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size  # Create real world coords. Use your metric.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.

    # Left directory path correction. Remove the last character if it is '/'
    if left_dir[-1:] == '/':
        left_dir = left_dir[:-1]

    # Right directory path correction. Remove the last character if it is '/'
    if right_dir[-1:] == '/':
        right_dir = right_dir[:-1]

    # Get images for left and right directory. Since we use prefix and formats, both image set can be in the same dir.
    left_images = glob.glob(left_dir + '/' + '*.' + image_format)
    #right_images = glob.glob(right_dir + '/' + '*.' + image_format)

    # Images should be perfect pairs. Otherwise all the calibration will be false.
    # Be sure that first cam and second cam images are correctly prefixed and numbers are ordered as pairs.
    # Sort will fix the globs to make sure.
    left_images.sort()

    print("Left images count: ", len(left_images))

    for left_im in left_images:
        base_name = os.path.basename(left_im)
        right_im = os.path.join(right_dir, base_name)
        if not os.path.exists(right_im):
            continue

        print('base_name:', base_name)
        # Right Object Points
        right = cv2.imread(right_im, 0)
        right = cv2.resize(right, (args.im_w, args.im_h))
        gray_right = right#cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
        # Left Object Points
        left = cv2.imread(left_im, 0)
        left = cv2.resize(left, (args.im_w, args.im_h))
        gray_left = left#cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:  # If both image is okay. Otherwise we explain which pair has a problem and continue
            #print('OK')
            show_right = cv2.drawChessboardCorners(right, (width, height), corners_right, ret_right)
            show_left = cv2.drawChessboardCorners(left, (width, height), corners_left, ret_left)
            cv2.imshow('stereo', np.hstack((show_left, show_right)))
            cv2.waitKey(100)
            # Object points
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_imgpoints.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(corners2_left)
        else:
            print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
            continue

    image_size = gray_right.shape  # If you have no acceptable pair, you may have an error here.
    return [objpoints, left_imgpoints, right_imgpoints]


if __name__ == '__main__':
    import os
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--left_file', type=str, required=True, help='left matrix file')
    parser.add_argument('--right_file', type=str, required=True, help='right matrix file')
    parser.add_argument('--left_prefix', type=str, required=False, help='left image prefix')
    parser.add_argument('--right_prefix', type=str, required=False, help='right image prefix')
    parser.add_argument('--left_dir', type=str, required=True, help='left images directory path')
    parser.add_argument('--right_dir', type=str, required=True, help='right images directory path')
    parser.add_argument('--image_format', type=str, default='jpg', help='image format, png/jpg')
    parser.add_argument('--width', type=int, default=9, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, default=6, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--square_size', type=float, default=1.0, required=False, help='chessboard square size')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save stereo calibration matrices')

    parser.add_argument('--im_h', type=int, default=1280)
    parser.add_argument('--im_w', type=int, default=720)

    args = parser.parse_args()
    save_dir = '/'.join(args.save_file.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # If chessboard pattern is different, we will pass them as arguments.
    if args.width is None and args.height is None:
        stereo_calibrate(args.left_file, args.right_file, args.left_dir, args.left_prefix, args.right_dir, args.right_prefix, args.image_format, args.save_file, args.square_size)
    else:
        stereo_calibrate(args.left_file, args.right_file, args.left_dir, args.left_prefix, args.right_dir, args.right_prefix, args.image_format, args.save_file, args.square_size, args.width, args.height)
