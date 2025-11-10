#include <iostream>
#include <opencv2/opencv.hpp> // Main OpenCV header
#include "orb.hpp" // Your ORB class header

int main()
{
    std::cout << "Starting ORB Demo..." << std::endl;

    // --- 1. Load a Test Image ---
    // Make sure you have a "test.jpg" image in your root ORB-Testing folder
    cv::Mat image = cv::imread("../Screenshot 2025-11-08 181210.png", cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cout << "Error: Could not load image" << std::endl;
        std::cout << "Please add a test.jpg to the ORB-Testing folder." << std::endl;
        return -1;
    }

    // --- 2. Initialize Your Class ---
    ORBDescriptor orb;

    // --- 3. Detect Keypoints ---
    std::vector<cv::KeyPoint> keypoints;
    keypoints = orb.detectKeypoints(image);
    std::cout << "Found " << keypoints.size() << " keypoints." << std::endl;

    // --- 4. Compute Orientation ---
    orb.computeOrientation(image, keypoints);

    // --- 5. Compute Descriptors ---
    cv::Mat descriptors;
    descriptors = orb.computeDescriptors(image, keypoints);
    std::cout << "Computed " << descriptors.rows << " descriptors (each " << descriptors.cols << " bytes)." << std::endl;

    // --- 6. Visualize the Results ---
    cv::Mat image_with_keypoints;
    cv::drawKeypoints(image, keypoints, image_with_keypoints, cv::Scalar(0, 255, 0), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("ORB Keypoints", image_with_keypoints);
    std::cout << "Displaying keypoints. Press any key to exit." << std::endl;

    cv::waitKey(0); // Wait for a key press

    return 0;
}