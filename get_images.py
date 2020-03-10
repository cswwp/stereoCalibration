import numpy as np
import cv2
import time
import sys
import os
# Set your camera

#0 left
#2 right
device_id = 2
cap = cv2.VideoCapture(device_id)




i = 0
width = 9
height = 6
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def main():
    global i
    if len(sys.argv) < 4:
        print("Usage: ./program_name directory_to_save start_index prefix")
        sys.exit(1)

    i = int(sys.argv[2])
    save_path = sys.argv[1]
    prefix = sys.argv[3]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print('size:', frame.shape)
        frame = cv2.resize(frame, (960, 540))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function.
            frame_point = cv2.drawChessboardCorners(frame.copy(), (width, height), corners2, ret)
            cv2.imshow('frame', frame_point)
        else:
            print('No cheese corner point find')

            # Display the resulting frame
            cv2.imshow('frame', frame)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(os.path.join(save_path, prefix + str(i) + ".png"), frame)
            i += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
