#include "orb.hpp"
#include <iostream>
#include <cmath>
#include <vector>

// Add required OpenCV includes
#include <opencv2/features2d.hpp> // For cv::FAST
#include <opencv2/imgproc.hpp>    // For cvRound

// This is the static data for the ORB pattern (typo corrected)
static const int bit_pattern_31_[512] =
{
    8, -3, 9, 5, 4, 2, 7, -12, 12, -13, 2, -13, 0, -11, -13, -13, -11, 8, -8, -13, -13, 11, -12, -8, -13, -3, -11, 6, -10, -1, -13, -1, 11, -1, 12, -6, 5, -3, 13, 12, -9, 0, 7, -5, 12, -11, 3, -10, -2, -13, 9, -13, 5, -13, -7, -13, -8, -8, -13, -7, -11, -13, -5, -10, 5, -7, -13, -2, -9, -13, -11, -13, 0, -13, 2, -12, -1, -10, -13, -10, -13, 0, -13, -5, -13, 7, -13, 11, -5, 5, -13, 8, -11, 9, -13, 2, -13, 7, -13, 1, -13, 3, -12, -2, -11, -13, -8, -13, 2, -13, -2, -13, -1, -13, -13, -13, -13, 5, -13, -10, -13, -13, -13, -13, -13, 8, -1, 11, -13, -13, -13, -13, -13, -13, -1, -11, 12, -11, 1, -10, 5, -10, 3, -8, -13, -8, -13, -2, -13, 8, -13, -11, 1, -13, -9, -13, -4, 1, -1, 1, -9, 3, -4, 6, -13, -12, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -11, -1, 4, 1, -13, -1, -13, 11, -13, -13, 2, -13, 4, -13, -1, -13, -10, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 1, -8, 2, -13, -13, -13, -13, -13, 4, -1, 12, 12, -13, -1, 2, -1, 12, -11, -10, -13, -13, -13, -13, -13, -13, -13, 9, -11, 11, -13, -13, -13, -13, -13, -13, -13, -13, -13, -10, -13, -13, -8, 8, -6, 12, -13, 5, -1, 13, 11, 0, -1, -13, -13, -13, -13, -13, -13, -13, 1, 12, 2, 9, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 10, -1, 12, 0, -13, -13, -13, -13, 3, -13, 5, -13, -13, -13, -13, -13, -13, -13, -13, -13, 7, -1, 11, 3, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 0, -10, 1, -13, -13, -13, -13, -13, 2, -4, 3, 2, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 4, -8, 4, -13, -13, -13, -13, -13, 9, -13, 11, -8, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13
};


ORBDescriptor::ORBDescriptor() {}

std::vector<cv::KeyPoint> ORBDescriptor::detectKeypoints(const cv::Mat& image)
{
    std::cout << "Detecting keypoints (stub)..." << std::endl;
    std::vector<cv::KeyPoint> keypoints;
    int threshold = 20; // Intensity difference threshold
    
    // Outsource to OpenCV
    cv::FAST(image, keypoints, threshold, true);
    
    return keypoints;
}

void ORBDescriptor::computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
    std::cout << "Computing orientation (refactored)..." << std::endl;

    const int patch_radius = 15; // patch around keypoint

    for (cv::KeyPoint& kp : keypoints)
    {
        // IMPORTANT: Initialize moments to zero for each keypoint
        double m00 = 0, m10 = 0, m01 = 0;
        
        // Call helper function to compute moments for the patch
        computePatchMoments(image, kp.pt, m00, m10, m01, patch_radius);

        if (m00 != 0)
        {
            // Calculate the angle from "desired orientation"
            double angle_rad = std::atan2(m01, m10);
            // Convert radians to degrees [0, 360]
            kp.angle = angle_rad * (180.0 / CV_PI);
            if (kp.angle < 0) {
                kp.angle += 360.0;
            }
        }
        else
        {
            // Handle case where m00 is zero (e.g., all black patch)
            kp.angle = 0; 
        }
    }
}

cv::Mat ORBDescriptor::computeDescriptors(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints)
{
    std::cout << "Computing descriptors (rotated BRIEF)..." << std::endl;

    // 32 bytes (256 bits) per descriptor
    cv::Mat descriptors(keypoints.size(), 32, CV_8U); 
    
    // Cast the static pattern to cv::Point for easier access
    const cv::Point* pattern = (const cv::Point*)bit_pattern_31_;

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const cv::KeyPoint& kp = keypoints[i];
        const cv::Point2f& center = kp.pt;

        // Get the pre-computed angle (in degrees) and convert to radians
        double angle_rad = kp.angle * (CV_PI / 180.0);
        double cos_theta = std::cos(angle_rad);
        double sin_theta = std::sin(angle_rad);

        // Get a pointer to the correct row in the output matrix
        uchar* desc_row = descriptors.ptr<uchar>(i);

        // Loop over all bytes
        for (int byte = 0; byte < 32; ++byte)
        {
            int byte_val = 0;
            
            // Loop over all 8 bits in the current byte
            for (int bit = 0; bit < 8; ++bit)
            {
                // Get the index for the current pair
                int pair_index = byte * 8 + bit;

                // Get the canonical coordinates from the pattern
                cv::Point p1_canon = pattern[pair_index];
                cv::Point p2_canon = pattern[pair_index + 256];

                // Rotate the pair by the keypoint's angle
                float x1_rot = p1_canon.x * cos_theta - p1_canon.y * sin_theta;
                float y1_rot = p1_canon.x * sin_theta + p1_canon.y * cos_theta;
                
                float x2_rot = p2_canon.x * cos_theta - p2_canon.y * sin_theta;
                float y2_rot = p2_canon.x * sin_theta + p2_canon.y * cos_theta;

                // Get the pixel locations
                int px1 = cvRound(center.x + x1_rot);
                int py1 = cvRound(center.y + y1_rot);
                
                int px2 = cvRound(center.x + x2_rot);
                int py2 = cvRound(center.y + y2_rot);

                // Get the intensity values (with boundary check)
                uchar val1 = 0, val2 = 0;
                if (py1 >= 0 && py1 < image.rows && px1 >= 0 && px1 < image.cols) {
                    val1 = image.at<uchar>(py1, px1);
                }
                if (py2 >= 0 && py2 < image.rows && px2 >= 0 && px2 < image.cols) {
                    val2 = image.at<uchar>(py2, px2);
                }

                // Binary test
                if (val1 < val2)
                {
                    // Set the bit in the byte
                    byte_val |= (1 << bit);
                }
            }
            
            // Store the completed byte
            desc_row[byte] = byte_val;
        }
    }

    return descriptors;
}

void ORBDescriptor::computePatchMoments(const cv::Mat& image, const cv::Point2f& center, 
                                        double& m00, double& m10, double& m01, int patch_radius)
{
    // Iterate over the patch
    // u is the column offset, v is the row offset
    for (int v = -patch_radius; v <= patch_radius; ++v)
    {
        for (int u = -patch_radius; u <= patch_radius; ++u)
        {
            // Calculate the pixel's absolute coordinates
            int col = cvRound(center.x + u);
            int row = cvRound(center.y + v);

            // Boundary Check
            if (row >= 0 && row < image.rows && col >= 0 && col < image.cols)
            {
                double intensity = image.at<uchar>(row, col); 
                
                m00 += intensity;
                m10 += u * intensity; // u is the x-offset from the center
                m01 += v * intensity; // v is the y-offset from the center
            }
        }
    }
}