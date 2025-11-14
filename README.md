# Visual Odometry
The goal of this project is to implement visual odometry with a monocular camera. By identifying keypoints, tracking them across multiple frames, and using that information to determine the camera's motion.

To do this, we started by utilizing existing codebases to create a full pipeline that executed visual odometry. We proceeded to implement custom FAST and BRIEF code for keypoint identification and feature extraction respectively.

## Using the Code

Our project relies on both Python and C++ to run, with the C++ code build using CMake. 
To run the code, ensure Python and CMake are both present on the system, as well as OpenCV for the C++ code. On Debian-based distributions, OpenCV can be installed with `sudo apt install libopencv-dev`.
Build the C++ code as follows:
```
mkdir build
cd build
cmake ..
make
cd ..
```
The C++ code can be tested by running the executable `./build/orb_demo`. If all works successfully, it should open the `/dev/video0` video device (typically the computer webcam) and draw keypoints on the image to show identified keypoints.

Our Python code also requires Pangolin (OpenGL library for visualization) be built and its Python bindings installed, which can be done following the instructions at https://github.com/uoip/pangolin. Other Python dependencies can be installed via the `requirements.txt` file with `pip install -r requirements.txt`. 

With the dependencies installed, the main pipeline can be run with `python main.py`. This, by default, operates on the test mp4 file at `third_part/test.mp4`, though the video source can be changed by passing in a different filepath to the cv.VideoCapture() constructor, or by passing an int `X` to signify the camera stream at `/dev/videoX`. This will launch two windows, one of which displays a point cloud of recognized and triangulated keypoints together with camera frustums every frame, and the other of which consists of two adjacent frames of the video stream with a small number of keypoints matches highlighted via connecting the keypoints on each frame to each other. 

Our pipeline can also be compared to the existing pipeline we built off of by running `python third_party/slam.py third_party/test.mp4`. Note that the scripts within `third_party/` are not ours, and were taken from the [https://github.com/Akbonline/SLAMPy-Monocular-SLAM-implementation-in-Python?tab=readme-ov-file](SLAMPy-Monocular-SLAM-implementation-in-Python) repository on GitHub, which did not include a license. Our changes to these scripts have been minimal and consisted of adding print statements for debugging purposes.

## Code Overview
## Algorithms
### FAST
FAST is an acronym for Features from Accelerated Segment Test. First proposed in the paper ____, FAST is designed as a computational efficient keypoint finding algorithm, compared to similar keypoint detection methods. This allows for a number of real-time applications, such as SLAM, to be signifigantly more viable.

The core of the FAST algorithm is that the algorithm considers each pixel in an image, comparing the pixel's intensity to a circle of surrounding pixels. If enough of the surrounding pixels are either consistently more or consistently less intense, the algorithm determines the pixel to be a keypoint.

We also implemented a couple improvements to this core logic: 
1. Only pixels whose circles will not go past the edge of the image will be considered. This is necessary to avoid undesired behavior from points that would be attempting to access pixel data that is outside of the camera frame.
2. Our algorithm first checks the pixels in the cardinal directions relative to the proposed keypoint pixel. This allows for pixels without __ to be quickly discarded in a fourth of the time.

### BRIEF
BRIEF is an acronym for Binary Robust Independent Elementary Features. First proposed in the paper "BRIEF: Binary Robust Independent Elementary Features", BRIEF takes in a list of keypoints and outputs matching bitstrngs designed to encode the features of each individual keypoint. Like FAST, BRIEF is designed to be computationally efficient, taking advantage of the low overhead required ot find the hamming distance between two bit strings allows for quick comparisons during keypoint matching.

The BRIEF algorithm starts by attempting to normalize the orientation of an image. This is done by calculating the centroid of an image. The centroid is made of up the moments of an image. Conceptually, the centroid represents ___ and the moments represent a weighted average of the images pixel intensities across a . These moments are calculated by 

## Challenges and Takeaways

In implementing certain parts of the visual odometry pipeline, we ran into a number of challenges and bugs, some of which we were able to resolve. After working past or around these, we also generated a list of takeaways and plans for how to better pursue a similar project in the future.

### Challenges

#### ORB

One of two major issues we overlooked when implementing ORB was not initially handling edge cases nicely when generating descriptors --- if we tried to sample a point outside the bounds of the image, we would move to the next keypoint in the loop, yet keep the previous descriptor at its default value of zero. This resulted in any keypoints within 16 pixels of the image boundary having an identical descriptor and erroneously matching with each other exactly. Integrating this with the matching visualization code was helpful in showing us this erroneous matching in action and letting us debug it.

Trying figure out why so many descriptors were zero, we realized we made another mistake, allocating only half as many ints as were needed to specify the X and Y coordinates of 256 different pairs of points, used by the BRIEF descriptor generation. This resulted in out-of-bounds reads for all our latter points in any pair, which, by chance, almost always gave us comparisons of zero, causing our descriptors to fail to convey any useful information. Once we realized this, we easily fixed it by doubling the number of ints in the uniform random int array.

One minor challenge was integrating the ORB implementation, written in C++, to the rest of our project in Python. As we hadn't recognized a benefit in using ROS2 or any other middleware that abstracts the C++-to-Python communcation, we ended up learning the basics of Pybind11 for building a Python interface into a C++ function, which is useful to know. 

#### Display

Rendering our point clouds and images proved to be consistently one of the most difficult parts of our implementation, given the different libraries and library wrappers used by the examples we found, as well as other assorted errors. Existing implementations we found used the Pangolin library for managing OpenGL display, which offers relatively easy support for rendering point clouds. However, two Python bindings for it exist, one of which could only be built with Python 3.10 or 3.9.

Additionally, we decided to use the Python wrapper for SDL2 for our own implementation given its versatility. However, when integrating the different components of our code, we consistently ran into segfaults when trying to access a SDL_Surface (and later, sometimes, SDL_Texture) object. This happened independent of our own C++-style code, and backtraces with GDB pointed to the segfault occurring from a function in the `libSDL2-2.0.so.0` library. The segfault would only occur when not running in the VSCode Python debugger, which made checking for null pointers difficult, and getting a more helpful backtrace from GDB would have required building Python and SDL2 with debug symbols, which we didn't think would be the most useful application of our time. The segfault would occur non-deterministically, and seemed tied to memory or some other hardware state --- on one occasion, we recorded 25 successful times running our program, and then a segfault on the 26th and a few subsequent times after that.

We've included some of the testing scripts we used for isolating what was causing the segfault and what wasn't in the `sdl2_test/` directory. Of these, `sdl2test.py` and `segfaults_not.py` never segfault, while `segfaults.py` does. While we tried debugging this further, the occurrence of segfaults was infrequent enough that we concluded they seem less present with SDL_Texture instead of SDL_Surface, and have left it at that. 

## Future Work

## Other notes

In addition to the included libraries, this project includes code from [https://github.com/Akbonline/SLAMPy-Monocular-SLAM-implementation-in-Python?tab=readme-ov-file](SLAMPy-Monocular-SLAM-implementation-in-Python), which does not list a license. All code taken from this repository is contained within the `third_party/` directory. 