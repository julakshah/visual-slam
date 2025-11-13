#include "orb.hpp"
#include <iostream>
#include <cmath>
#include <vector>

// Add required OpenCV includes
#include <opencv2/features2d.hpp> // For cv::FAST
#include <opencv2/imgproc.hpp>    // For cvRound

// Static data for the ORB pattern
static const int bit_pattern_31_[512] =
{
    8, -3, 9, 5, 4, 2, 7, -12, 12, -13, 2, -13, 0, -11, -13, -13, -11, 8, -8, -13, -13, 11, -12, -8, -13, -3, -11, 6, -10, -1, -13, -1, 11, -1, 12, -6, 5, -3, 13, 12, -9, 0, 7, -5, 12, -11, 3, -10, -2, -13, 9, -13, 5, -13, -7, -13, -8, -8, -13, -7, -11, -13, -5, -10, 5, -7, -13, -2, -9, -13, -11, -13, 0, -13, 2, -12, -1, -10, -13, -10, -13, 0, -13, -5, -13, 7, -13, 11, -5, 5, -13, 8, -11, 9, -13, 2, -13, 7, -13, 1, -13, 3, -12, -2, -11, -13, -8, -13, 2, -13, -2, -13, -1, -13, -13, -13, -13, 5, -13, -10, -13, -13, -13, -13, -13, 8, -1, 11, -13, -13, -13, -13, -13, -13, -1, -11, 12, -11, 1, -10, 5, -10, 3, -8, -13, -8, -13, -2, -13, 8, -13, -11, 1, -13, -9, -13, -4, 1, -1, 1, -9, 3, -4, 6, -13, -12, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -11, -1, 4, 1, -13, -1, -13, 11, -13, -13, 2, -13, 4, -13, -1, -13, -10, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 1, -8, 2, -13, -13, -13, -13, -13, 4, -1, 12, 12, -13, -1, 2, -1, 12, -11, -10, -13, -13, -13, -13, -13, -13, -13, 9, -11, 11, -13, -13, -13, -13, -13, -13, -13, -13, -13, -10, -13, -13, -8, 8, -6, 12, -13, 5, -1, 13, 11, 0, -1, -13, -13, -13, -13, -13, -13, -13, 1, 12, 2, 9, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 10, -1, 12, 0, -13, -13, -13, -13, 3, -13, 5, -13, -13, -13, -13, -13, -13, -13, -13, -13, 7, -1, 11, 3, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 0, -10, 1, -13, -13, -13, -13, -13, 2, -4, 3, 2, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, 4, -8, 4, -13, -13, -13, -13, -13, 9, -13, 11, -8, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13
};

static const int bit_pattern_31_new[1024] = 
{
    -15,-6,-2,-5,-10,11,13,10,-5,9,-2,13,11,-10,9,10,-13,-13,3,13,-9,1,10,-11,3,10,7,7,8,13,14,10,-7,-2,-13,9,-6,3,2,3,-9,-4,-5,0,-13,14,-15,-15,11,-6,15,13,-1,0,0,6,3,6,14,14,2,8,-2,-4,3,7,-6,7,12,-4,14,0,12,-1,0,-14,7,4,7,-8,-5,-13,7,-15,2,-7,-7,-4,-10,-12,-2,11,2,-4,-12,3,14,-13,15,-5,4,2,-11,13,-1,10,-11,2,14,14,0,8,4,-7,-10,-12,0,-8,-14,-6,15,1,-11,-4,-2,14,0,-9,13,9,11,-7,12,15,-2,2,8,10,15,-9,6,-1,-2,-8,-15,-13,-6,13,-11,-6,10,1,13,-10,-14,-13,-10,-3,9,1,8,-5,3,-10,-9,-11,10,-1,2,0,-10,11,13,-11,-15,0,-6,0,2,12,10,-1,-14,-1,14,14,-12,-12,0,-5,-2,-13,-7,15,-5,-8,6,4,-9,7,-14,-10,6,14,8,-6,1,13,-15,-6,-7,8,7,-12,14,3,-4,-3,-3,10,5,-4,-1,14,6,9,-5,-7,-10,13,15,-6,1,3,-9,-8,13,15,9,4,15,13,-8,-13,15,5,0,1,-5,-1,-6,1,9,9,14,1,-8,1,-3,-5,8,0,-11,-9,9,12,-3,5,-10,-5,11,-6,-11,-14,-15,-5,-14,14,-6,4,9,-5,8,-2,-11,9,7,5,4,-3,-1,0,-10,-11,-13,14,-12,10,0,-1,-2,-14,15,-4,0,13,-5,2,3,6,-4,-11,-3,0,14,-8,-4,-5,3,12,-7,-15,4,4,12,-14,1,0,7,9,-10,3,1,2,-12,-2,-12,8,8,-2,7,8,-9,-8,-5,-3,14,2,-11,3,5,9,-13,-4,-9,14,-6,8,7,7,-1,-1,8,2,-6,-12,-4,-7,-5,5,1,9,4,4,6,-8,5,-9,-9,-11,-6,-9,15,7,5,3,-2,0,5,-8,7,-3,-15,-15,-6,-6,-5,-10,-8,-15,-13,9,-11,8,-10,-10,6,-2,7,-14,10,-8,2,7,-3,-6,4,13,2,4,-6,-5,10,-4,-13,8,12,-8,-11,-11,14,13,3,-9,9,-13,2,0,-8,5,6,-3,9,-15,-10,8,-3,2,11,1,-10,3,1,12,5,13,14,6,-11,12,-12,15,1,9,15,14,-5,-12,-1,-13,3,-7,4,2,-10,2,0,-6,-1,-1,-8,1,3,-2,-12,-5,-4,-8,-12,8,2,-6,0,-12,-11,10,15,8,15,1,-9,-15,8,0,1,-9,6,10,0,12,10,-7,-6,10,-10,-1,-3,-11,-10,2,-3,-5,-3,-4,-11,12,-15,6,13,-12,10,-1,-3,2,11,-1,0,-10,15,8,2,-5,9,-15,4,3,-4,-2,1,-13,10,-6,-6,-6,-15,10,-6,-9,15,5,-8,-13,9,8,-10,-3,-13,-7,-1,-8,15,-8,11,-2,6,-12,-5,9,8,14,8,2,-10,2,11,5,13,13,-12,13,-14,9,-6,-14,11,-12,14,-7,-3,-6,3,7,2,-5,2,-12,1,4,1,10,10,5,-3,-2,15,-1,4,-3,12,-13,-11,-7,-14,-4,12,9,-6,1,4,11,-5,-13,6,-3,-11,-2,1,-13,3,4,-8,3,-14,-14,-12,-7,9,-10,1,7,15,6,-8,1,-5,3,13,-9,-5,1,9,0,-12,11,-10,8,13,13,-15,12,3,-2,-3,5,-9,3,-3,-3,-5,-8,3,-6,14,-5,-9,-8,-9,-3,10,13,12,0,4,-4,7,-8,-8,-12,-9,12,8,15,12,13,-4,9,14,-13,11,4,-6,-4,6,15,-10,-15,-7,-10,10,-3,0,1,-7,12,4,12,3,15,8,12,9,-9,9,3,-2,-7,12,-5,13,-6,0,14,-1,7,-5,-14,-6,-12,4,-7,15,-15,5,3,4,5,-9,0,-4,9,-9,14,4,1,7,-1,-9,-2,-11,-13,-6,-6,-15,-9,14,-2,3,-5,3,-9,7,2,-5,0,8,-12,10,7,-8,-2,1,-12,5,11,-4,-12,-11,-9,3,13,-1,12,6,0,-7,-15,-1,2,1,-3,6,-5,10,-5,9,-13,-4,8,-1,14,-9,3,14,-7,-1,8,-2,-3,-8,-15,-15,2,-15,1,-3,11,-12,2,5,-2,11,-4,-12,-11,-3,-4,1,-14,-2,8,-3,-11,-3,4,-12,6,-12,13,-10,5,8,-7,-5,15,8,-8,-3,9,8,-14,-13,4,13,-9,10,6,6,4,-7,4,-5,14,-5,14,5,-9,-5,-15,-12,13,1,-5,14,11,-12,-3,-7,1,-2,-3,12,-14,10,13,1,-10,11,-2,-10,-11,-7,11,5,-13,15,2,14,7,-12,2,-7,-4,2,-4,1,-2,-4,12,-2,3,11,8,15,-10,-1,10,-1,-12,7,-14,13,-11,5,5,-13,-9,-6,11,5,-9,-2,-5,-12,12,-9,-10,2,14,12,15,0,-7,-2,-3,-13,-13,5,11,13,-3,3,-8,5,6,-2,13,-11,1,2,14,-2,-2,5,-15,-5,-4,1,0,-14,14,12,-15,-15,-5,14,-8,13,15,-8,7,10,12,14,15,-3,9,5,12,-4,-15,-6,11,6,-2,-12,4,15,0,-15,-14,-13,-11,7,-7
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
        // Initialize moments to zero for each keypoint
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
    std::cout << "Computing descriptors ..." << std::endl;

    // 32 bytes (256 bits) per descriptor
    // cv::Mat descriptors(keypoints.size(), 32, CV_8U);
    const int PATCH_RADIUS = 16;

    // Filter out keypoints too close to edges
    std::vector<int> valid_keypt_idx;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        const cv::KeyPoint& kp = keypoints[i];
        const cv::Point2f& center = kp.pt;
    
        if (center.y - PATCH_RADIUS >= 0 && center.y + PATCH_RADIUS < image.rows && center.x - PATCH_RADIUS >= 0 && center.x + PATCH_RADIUS < image.cols) {
            valid_keypt_idx.push_back(i);
        }
    }

    cv::Mat good_desc(valid_keypt_idx.size(), 32, CV_8U);
    
    // Cast the static pattern to cv::Point for easier access
    const cv::Point* pattern = (const cv::Point*)bit_pattern_31_new;

    for (size_t i = 0; i < valid_keypt_idx.size(); ++i)
    {
        const cv::KeyPoint& kp = keypoints[valid_keypt_idx[i]];
        const cv::Point2f& center = kp.pt;

        // Get the pre-computed angle (in degrees) and convert to radians
        double angle_rad = kp.angle * (CV_PI / 180.0);
        double cos_theta = std::cos(angle_rad);
        double sin_theta = std::sin(angle_rad);

        // Get a pointer to the correct row in the output matrix
        uchar* desc_row = good_desc.ptr<uchar>(i);

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

                //std::cout << "px1, py1: " << px1 << "  " << py1 << ",  px2, py2: " << px2 << "  " << py2 << std::endl;

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
            // std::cout << byte_val << ", ";
        }
    }

    return good_desc;
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