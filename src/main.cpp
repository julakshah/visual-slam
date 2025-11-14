// #include <iostream>
// #include <sstream>    // For creating text strings
// #include <iomanip>    // For formatting text (std::setprecision)
// #include <opencv2/opencv.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/features2d.hpp> // For cv::ORB and cv::BFMatcher
// #include "orb.hpp"                // my ORB class header

// int main()
// {
//     // --- 1. INITIALIZE ---
//     std::cout << "Starting live ORB verification..." << std::endl;
//     std::cout << "Press 'q' to quit." << std::endl;

//     // My implementation
//     ORBDescriptor my_orb;

//     // OpenCV's implementation (for comparison)
//     cv::Ptr<cv::ORB> cv_orb = cv::ORB::create();

//     // Brute-Force Matcher (do this once)
//     cv::BFMatcher matcher(cv::NORM_HAMMING);

//     // Webcam
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened())
//     {
//         std::cout << "Error: Could not open webcam." << std::endl;
//         return -1;
//     }

//     // --- 2. START VIDEO LOOP ---
//     cv::Mat frame;
//     cv::Mat gray_frame;

//     while (true)
//     {
//         cap.read(frame);
//         if (frame.empty())
//         {
//             std::cout << "Error: Grabbed empty frame." << std::endl;
//             break;
//         }

//         cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

//         // --- 3. DETECT KEYPOINTS ---
//         // Get one common set of keypoints using my FAST stub
//         std::vector<cv::KeyPoint> common_keypoints = my_orb.detectKeypoints(gray_frame);
        
//         // --- 4. RUN BOTH IMPLEMENTATIONS ---
        
//         // A) my Implementation
//         std::vector<cv::KeyPoint> my_keypoints = common_keypoints;
//         cv::Mat my_descriptors;
//         if (!my_keypoints.empty())
//         {
//             my_orb.computeOrientation(gray_frame, my_keypoints);
//             my_descriptors = my_orb.computeDescriptors(gray_frame, my_keypoints);
//         }

//         // B) OpenCV's Implementation
//         std::vector<cv::KeyPoint> cv_keypoints = common_keypoints;
//         cv::Mat cv_descriptors;
//         if (!cv_keypoints.empty())
//         {
//             cv_orb->compute(gray_frame, cv_keypoints, cv_descriptors);
//         }

//         // --- 5. REAL-TIME MATCHING ---
//         int num_good_matches = 0;
//         float match_rate = 0.0f;

//         if (my_descriptors.empty() || cv_descriptors.empty())
//         {
//             // Can't match if one set is empty (e.g., dark frame)
//         }
//         else
//         {
//             std::vector<std::vector<cv::DMatch>> knn_matches;
//             matcher.knnMatch(my_descriptors, cv_descriptors, knn_matches, 2);

//             // Filter matches using Lowe's Ratio Test
//             const float ratio_thresh = 0.75f;
//             for (size_t i = 0; i < knn_matches.size(); i++)
//             {
//                 if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < knn_matches[i][1].distance * ratio_thresh)
//                 {
//                     num_good_matches++;
//                 }
//             }
            
//             if (my_descriptors.rows > 0)
//             {
//                 match_rate = (float)num_good_matches / my_descriptors.rows * 100.0f;
//             }
//         }
        
//         // --- 6. VISUALIZE ---
        
//         // A) Draw my keypoints (with my orientation)
//         cv::Mat output_frame;
//         cv::drawKeypoints(frame, my_keypoints, output_frame, cv::Scalar(0, 255, 0), 
//                           cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//         // B) Draw the verification text
//         std::string kp_text = "Keypoints: " + std::to_string(common_keypoints.size());
//         std::string match_text = "Good Matches: " + std::to_string(num_good_matches);
        
//         std::ostringstream ss;
//         ss << "Match Rate: " << std::fixed << std::setprecision(2) << match_rate << "%";
//         std::string rate_text = ss.str();

//         cv::putText(output_frame, kp_text, cv::Point(10, 30), 
//                     cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
//         cv::putText(output_frame, match_text, cv::Point(10, 60), 
//                     cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
//         cv::putText(output_frame, rate_text, cv::Point(10, 90), 
//                     cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

//         // C) Show the frame
//         cv::imshow("Live ORB Verification (my Code vs OpenCV)", output_frame);

//         // --- 7. QUIT ---
//         if (cv::waitKey(1) == 'q')
//         {
//             break;
//         }
//     }

//     // --- 8. CLEAN UP ---
//     cap.release();
//     cv::destroyAllWindows();
//     std::cout << "Webcam feed stopped." << std::endl;

//     return 0;
// }



#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <chrono>
#include <vector>

// Struct to hold results
struct FeatureResult {
    int num_kp1;
    int num_kp2;
    int num_matches;
    double time_seconds;
};

// Detect, compute, match
FeatureResult testFeatures(const cv::Mat& img1, const cv::Mat& img2,
                           cv::Ptr<cv::Feature2D> detector,
                           cv::Ptr<cv::Feature2D> descriptor = nullptr)
{
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;

    auto start = std::chrono::high_resolution_clock::now();

    // Detect keypoints
    detector->detect(img1, kp1);
    detector->detect(img2, kp2);

    // Compute descriptors if provided
    if (descriptor) {
        descriptor->compute(img1, kp1, des1);
        descriptor->compute(img2, kp2, des2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    int num_matches = 0;
    if (!des1.empty() && !des2.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        std::vector<cv::DMatch> matches;
        matcher.match(des1, des2, matches);
        num_matches = static_cast<int>(matches.size());
    }

    return { static_cast<int>(kp1.size()), static_cast<int>(kp2.size()), num_matches, elapsed };
}

int main() {
    // --- Load images ---
    std::string path1 = "path_to_hpatches/sequence_name/1.ppm"; // Replace with your image
    std::string path2 = "path_to_hpatches/sequence_name/2.ppm";

    cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error loading images!\n";
        return -1;
    }

    // --- ORB ---
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    FeatureResult res_orb = testFeatures(img1, img2, orb, orb);
    std::cout << "ORB: " << res_orb.num_kp1 << "+" << res_orb.num_kp2
              << " keypoints, " << res_orb.num_matches
              << " matches, " << res_orb.time_seconds << "s\n";

    // --- FAST + BRIEF ---
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(25, true);
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief =
        cv::xfeatures2d::BriefDescriptorExtractor::create(32);

    FeatureResult res_fb = testFeatures(img1, img2, fast, brief);
    std::cout << "FAST+BRIEF: " << res_fb.num_kp1 << "+" << res_fb.num_kp2
              << " keypoints, " << res_fb.num_matches
              << " matches, " << res_fb.time_seconds << "s\n";

    return 0;
}
