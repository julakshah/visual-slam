#ifndef ORB_HPP
#define ORB_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/features2d.hpp>

class ORBDescriptor
{
public:
    ORBDescriptor();

    // Detect keypoints (FAST)
    static std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& image);

    // Compute orientation using intensity moments
    static void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

    // Compute binary descriptor
    static cv::Mat computeDescriptors(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);

private:
    // Helper for computing raw moments in a patch
    static void computePatchMoments(const cv::Mat& image, const cv::Point2f& center, 
                             double& m00, double& m10, double& m01, int patch_radius);
};

#endif // ORB_HPP