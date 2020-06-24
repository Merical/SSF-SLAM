# SSF-SLAM

A Self-Supervised-Feature-Slam System with RGB-D Camera.
The Feature-based VSLAM system with self-supervised feature detection is referring to [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2).

## Prerequisites

### System

Ubuntu16.04

### C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

### CUDA&CUDNN
Cuda 10.* and CUDNN are required for Feature Detection Network Inference.

### Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### OpenCV

We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 3.4.0. Tested with OpenCV 3.4**.

### Eigen3

Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

### DBoW3 and g2o (Included in Thirdparty folder)

We use modified versions of the [DBoW3](https://github.com/rmsalinas/DBow3) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

### LibTorch
[LibTorch](https://pytorch.org/) is required for Feature Detection Network Inference. 
\
**Tested under v1.3.1**

## Build
### Clone the repository
```shell script
git clone https://github.com/Merical/self-supervised-feature-slam.git
```

### Build the project
```shell script
cd SSF-SLAM
chmod +x build.sh
./build.sh
```
This will the executables create **rgbd_lch** in LCHP folder.

## Examples
### Associate the sequence
```shell script
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```

### Execute
```shell script
cd LCHP
./rgbd_lchp path_to_vocabulary path_to_settings path_to_sequence path_to_association path_to_trajectory_dir
```
