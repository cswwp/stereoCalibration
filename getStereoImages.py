import numpy as np
import cv2
import glob
import argparse
import sys
import os

# Set the values for your cameras
capL = cv2.VideoCapture(0) #left
capR = cv2.VideoCapture(2) #right

# Use these if you need high resolution.
# capL.set(3, 1024) # width
# capL.set(4, 768) # height

# capR.set(3, 1024) # width
# capR.set(4, 768) # height
i = 0
width=9
height=6
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():
    global i
    if len(sys.argv) < 3:
        print("Usage: ./program_name directory_to_save start_index")
        sys.exit(1)

    i = int(sys.argv[2])  # Get the start number.

    save_path = sys.argv[1]
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    while True:
        # Grab and retreive for sync
        if not (capL.grab() and capR.grab()):
            print("No more frames")
            break

        _, leftFrame = capL.retrieve()
        _, rightFrame = capR.retrieve()

        leftFrame = cv2.resize(leftFrame, (960, 540))
        rightFrame = cv2.resize(rightFrame, (960, 540))

        leftgray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        rightgray = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        leftret, leftcorners = cv2.findChessboardCorners(leftgray, (width, height), None)
        rightret, rightcorners = cv2.findChessboardCorners(rightgray, (width, height), None)

        if leftret and rightret:
            leftcorner2 = cv2.cornerSubPix(leftgray, leftcorners, (11, 11), (-1, -1), criteria)
            rightcorner2 = cv2.cornerSubPix(rightgray, rightcorners, (11, 11), (-1, -1), criteria)

            left_show = cv2.drawChessboardCorners(leftFrame.copy(), (width, height), leftcorner2, leftret)
            right_show = cv2.drawChessboardCorners(rightFrame.copy(), (width, height), rightcorner2, rightret)

            cv2.imshow('frame', np.hstack((left_show, right_show)))
        else:
            print('No cheese point detect')
            cv2.imshow('frame', np.hstack((leftFrame, rightFrame)))

        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(os.path.join(save_path, "left" + str(i) + ".png"), leftFrame)
            cv2.imwrite(os.path.join(save_path, "right" + str(i) + ".png"), rightFrame)
            i += 1

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
