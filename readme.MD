# Stereo calibration and disparity creation



This project is a basis for stereo vision.  After calibration process, you can try to get disparity map and start your own project. 

## Usage
### Single camera calibration

Example usage:


python single_camera_calibration.py
-- image_dir 拍摄的棋格图片所在位置， 例如左摄像头 /image_transform/image0

--image_format 图片格式，默认jpg
--square_size 不需要管
--prefix 不需要管
--width 棋盘竖直方向角点个数
--height 棋盘水平方向角点个数
--save_file 当前标定的摄像机内参保存路径。规范统一格式为：保存目录/left.yml  或者保存目录/right.yml，如果你标定的是image0里边的图片，就选left.yml, 反之选right.yml





### Stereo camera calibration

Example usage:

python stereo_camera_calibration.py
--left_file 已经标好的做相机的参数, 例如 left.yml
--right_file已经标好的做相机的参数, 例如 right.yml
--image_dir 所要标定的图片所在位置， 例如 image_transform/，  子目录为image0 image1
--left_prefix 不需要管
--right_prefix 不需要管
--left_dir 左图目录， 例如image_trnasform/image0
--right_dir 右图目录， 例如image_trnasform/image1
--image_format 图片格式，默认jpg
--width 棋盘竖直方向角点个数
--height 棋盘水平方向角点个数
--square_size 不需要管
--save_file 双目标定参数存放的位置  统一格式为：保存目录/stereo.yml



### Disparity map 测试

Example usage:

python stereo940.py
--test_dir 你的测试图所在目录， 子目录为image0 image1
--calibration_file 你标定出的双目参数stere.yml 位置
--height 图像高度
--width 图像宽度



总结下来：
拍摄棋格纸：9x6角点

拍摄要求: 30炒米， 50炒米，100m， 三个距离

将相机成像显示的图片划分为4个象限每个象限进行拍摄





①左右两个相机分别针对已拍摄好的各自的单目图片（来源于1）进行标定， 脚本single_camera_calibration.py， 自动保存参数到给定的命令行参数位置

②双目标定，图片来源于2， 脚本stereo_camera_calibration.py， 自动保存相机参数，自行查看参数

③用3，4获得的参数，计算深度图， 运行stere940.py, 指定自己测试图所在的文件夹


需要自己看下脚本中的参数， 方便自己使用
注意事项：
1 相机左右分清楚， 当你面向相机的时候，你右手的相机是左相机， 你左手边的相机是右相机， 现在已经固定死了，image0 是left摄像机， image1是右摄像机

2 一定要自行查看命令行参数


## 样例
640x360尺度下的标定 （高640， 宽320）

①标定 image0 左摄像头
python single_camera_calibration.py --image_dir sample/image0/ --save_file local_params_640x360/left.yml --im_w 360 --im_h 640

②标定 image1 右摄像头
python single_camera_calibration.py --image_dir sample/image1/ --save_file local_params_640x360/right.yml --im_w 360 --im_h 640

③立体矫正
python stereo_camera_calibration.py --left_file local_params_640x360/left.yml --right_file local_params_640x360/right.yml --left_dir sample/image0/ --right_dir sample/image1/ --save_file local_params_640x360/stereo.yml --im_w 360 --im_h 640

④测试结果
python stereo940.py --calibration_file local_params_640x360/stereo.yml --im_h 640 --im_w 320 --test_dir test_sample940_1/







1280x720尺度下的标定（高1280， 宽720）

①标定 image0 左摄像头
python single_camera_calibration.py --image_dir sample/image0/ --save_file local_params_1280x720/left.yml --im_w 720 --im_h 1280

②标定 image1 右摄像头
python single_camera_calibration.py --image_dir sample/image1/ --save_file local_params_1280x720/right.yml --im_w 720 --im_h 1280

③立体矫正
python stereo_camera_calibration.py --left_file local_params_1280x720/left.yml --right_file local_params_1280x720/right.yml --left_dir sample/image0/ --right_dir sample/image1/ --save_file local_params_1280x720/stereo.yml --im_w 720 --im_h 1280

④测试结果
python stereo940.py --calibration_file local_params_640x360/stereo.yml --im_h 1280 --im_w 720 --test_dir test_sample940_1/








