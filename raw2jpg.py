import numpy as np
import cv2
import glob
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

raw_shape = (720, 1280)

debug_show = 1
debug_show_slide = 0

pattern_size = (9, 6)
im_h = 640
im_w = 360



def quantization(matin):
    # quantized uint8
    min_z = float(np.min(matin))
    max_z = float(np.max(matin))
    return np.flipud(((matin - min_z) / (max_z-min_z) * 255).astype(np.uint8).transpose())
    

def imread_from_raw_left(raw_file):    
    # read raw data
    raw_shape_left = raw_shape + (2,)
    raw = np.fromfile(raw_file, dtype=np.uint8).reshape(raw_shape_left).astype(np.uint16)
    # merge channels
    gray_short = np.bitwise_or(np.bitwise_and(np.left_shift(raw[:,:,0], 8), 0xff00), 
                               np.right_shift(np.bitwise_and(raw[:,:,1], 0x00f0), 4))
   
    return quantization(gray_short)

def imread_from_raw_right(raw_file):
    raw = np.fromfile(raw_file, dtype=np.uint16).reshape(raw_shape)
    return quantization(raw)


def merge_binocular_data(indir, outdir):
    indir0_pattern = os.path.join(indir, "image0", "*.raw")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename0 in glob.glob(indir0_pattern):
        print(filename0)
        basename = os.path.basename(filename0)
        filename1 = os.path.join(indir, "image1", basename)
        if os.path.exists(filename1):
            left = imread_from_raw_left(filename0)
            right = imread_from_raw_right(filename1)

            merged = np.hstack((left, right))
            if debug_show:
                mshape = merged.shape
                # for easy show
                resized = cv2.resize(merged, (int(mshape[0]/2), int(mshape[1]/2)))
                cv2.imshow("merged", resized)
                cv2.waitKey(debug_show_slide)
            outfile = os.path.join(outdir, basename+".png")
            cv2.imwrite(outfile, merged)

def save_mono_cheese(indir, outdir, sign):
    if sign: #left
        indir_pattern = os.listdir(os.path.join(indir, 'image0'))#os.path.join(indir, "image0", "*.raw")
        sub_dir = os.path.join(indir, 'image0')
    else:
        indir_pattern = os.listdir(os.path.join(indir, "image1"))
        sub_dir = os.path.join(indir, 'image1')
    indir_pattern = [os.path.join(sub_dir, fn) for fn in indir_pattern]

    #print(type(indir_pattern))
    indir_pattern = sorted(indir_pattern)
    sub_dir = 'image0' if sign else 'image1'
    savepath = os.path.join(outdir, sub_dir)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for filename in indir_pattern:
        basename = os.path.basename(filename)
        print('basename:', basename)
        if sign:
            img = imread_from_raw_left(filename)
        else:
            img = imread_from_raw_right(filename)

        img_small = cv2.resize(img, (im_w, im_h))


        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_small, pattern_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(img_small, corners, (11, 11), (-1, -1), criteria)
            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function.
            frame_point = cv2.drawChessboardCorners(img_small.copy(), pattern_size, corners2, ret)
            cv2.imshow('frame', frame_point)

            if cv2.waitKey(0) & 0xFF == ord('c'):
                print('sfdasF:', os.path.join(savepath, basename))
                cv2.imwrite(os.path.join(savepath, basename.replace('.raw', '.jpg')), img)




def trans_sample(indir, outdir):
    indir0_pattern = os.path.join(indir, "image0", "*.raw")

    out_dir0 = os.path.join(outdir, 'image0')
    out_dir1 = os.path.join(outdir, 'image1')
    if not os.path.exists(out_dir0):
        os.makedirs(out_dir0)
    if not os.path.exists(out_dir1):
        os.makedirs(out_dir1)


    for filename0 in glob.glob(indir0_pattern):
        print(filename0)
        basename = os.path.basename(filename0)
        filename1 = os.path.join(indir, "image1", basename)
        if os.path.exists(filename1):
            left = imread_from_raw_left(filename0)
            right = imread_from_raw_right(filename1)
            cv2.imshow("merged", cv2.resize(np.hstack((left, right)), (360, 640)))
            if cv2.waitKey(0) & 0xFF == ord('c'):
                cv2.imwrite(os.path.join(out_dir0, basename + '.jpg'), left)
                cv2.imwrite(os.path.join(out_dir1, basename + '.jpg'), right)










if __name__ == "__main__":
    # #merge_binocular_data("./raw", "./merged")
    save_mono_cheese('raw', 'sample', 1)
    save_mono_cheese('raw', 'sample', 0)

    # trans_sample('raw_test_sample', 'test_sample940_1')






