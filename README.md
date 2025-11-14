# Visual Odometry
The goal of this project is to implement visual odometry with a monocular camera. By identifying keypoints, tracking them across multiple frames, and using that information to determine the camera's motion.

To do this, we started by utilizing existing codebases to create a full pipeline that executed visual odometry. We proceeded to implement custom FAST and BRIEF code for keypoint identification and feature extraction respectively.
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
## Future Work
