#ifndef ORB_HPP
#define ORB_HPP

#include <opencv2/core.hpp>

class ORBDescriptor
{
public:
    ORBDescriptor();

    // Detect keypoints (FAST)
    std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& image);

    // Compute orientation using intensity moments
    void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

    // Compute binary descriptor
    cv::Mat computeDescriptors(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);

private:
    // Helper for computing raw moments in a patch
    void computePatchMoments(const cv::Mat& patch, double& m00, double& m10, double& m01);
};

#endif // ORB_HPP
