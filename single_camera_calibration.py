import numpy as np
import cv2
import glob
import argparse
from calibration_store import save_coefficients

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size  # Create real world coords. Use your metric.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Directory path correction. Remove the last character if it is '/'
    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    # Get the images
    images = glob.glob(dirpath+'/' + '*.' + image_format)


    # Iterate through the pairs and find chessboard corners. Add them to arrays
    # If openCV can't find the corners in an image, we discard the image.
    for fname in images:
        img = cv2.imread(fname, 0)
        img = cv2.resize(img, (args.im_w, args.im_h))
        gray = img

        #gray = cv2.warpAffine(gray, M, (height, width))  # 13

        # gray = np.rot90(gray)
        # cv2.imshow('gray', gray)

        #cv2.waitKey(0)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print('gray:', gray.shape)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        #print(ret, corners)
        # If found, add object points, image points (after refining them)
        if ret:
            print('ok')
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function.
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow('cheese', img)
            cv2.waitKey(10)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    import os
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, default='jpg',  help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=False, help='image prefix')
    parser.add_argument('--square_size', type=float, default=1.0,required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, default=9, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, default=6, help='chessboard height size, default is 6')
    parser.add_argument('--im_h', type=int, default=1280)
    parser.add_argument('--im_w', type=int, default=720)
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()

    # Call the calibraton and save as file. RMS is the error rate, it is better if rms is less than 0.2
    ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.prefix, args.image_format, args.square_size, args.width, args.height)

    save_dir = '/'.join(args.save_file.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_coefficients(mtx, dist, args.save_file)
    print("Calibration is finished. RMS: ", ret)
