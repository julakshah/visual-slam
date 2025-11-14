// This file is used to compare our ORB implementation against OpenCV's

#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include "orb.hpp"

int main()
{
    std::cout << "Starting: Press 'q' to quit." << std::endl;

    // Our implementation
    ORBDescriptor my_orb;

    // OpenCV's implementation
    cv::Ptr<cv::ORB> cv_orb = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat gray_frame;

    while (true)
    {
        cap.read(frame);
        if (frame.empty())
        {
            std::cout << "Error: Grabbed empty frame." << std::endl;
            break;
        }

        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

        // Detecting keypoints (use same keypoints for both implementations)
        std::vector<cv::KeyPoint> common_keypoints = my_orb.detectKeypoints(gray_frame);

        // Our Implementation
        std::vector<cv::KeyPoint> my_keypoints = common_keypoints;
        cv::Mat my_descriptors;
        if (!my_keypoints.empty())
        {
            my_orb.computeOrientation(gray_frame, my_keypoints);
            my_descriptors = my_orb.computeDescriptors(gray_frame, my_keypoints);
        }

        // OpenCV's Implementation
        std::vector<cv::KeyPoint> cv_keypoints = common_keypoints;
        cv::Mat cv_descriptors;
        if (!cv_keypoints.empty())
        {
            cv_orb->compute(gray_frame, cv_keypoints, cv_descriptors);
        }

        // Find matches
        int num_good_matches = 0;
        float match_rate = 0.0f;

        if (my_descriptors.empty() || cv_descriptors.empty())
        {
            // Handle case with no descriptors
        }
        else
        {
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(my_descriptors, cv_descriptors, knn_matches, 2);

            // Filter matches using Lowe's Ratio Test
            const float ratio_thresh = 0.75f;
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < knn_matches[i][1].distance * ratio_thresh)
                {
                    num_good_matches++;
                }
            }
            
            if (my_descriptors.rows > 0)
            {
                match_rate = (float)num_good_matches / my_descriptors.rows * 100.0f;
            }
        }
        
        // Visuals
        
        // Draw our keypoints
        cv::Mat output_frame;
        cv::drawKeypoints(frame, my_keypoints, output_frame, cv::Scalar(0, 255, 0), 
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Draw the verification text
        std::string kp_text = "Keypoints: " + std::to_string(common_keypoints.size());
        std::string match_text = "Our/OpenCV Matches: " + std::to_string(num_good_matches);
        
        std::ostringstream ss;
        ss << "Match Rate: " << std::fixed << std::setprecision(2) << match_rate << "%";
        std::string rate_text = ss.str();

        cv::putText(output_frame, kp_text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(output_frame, match_text, cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(output_frame, rate_text, cv::Point(10, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Live ORB Verification (my Code vs OpenCV)", output_frame);

        // Exit condition
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Webcam feed stopped." << std::endl;

    return 0;
}