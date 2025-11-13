#include <iostream>
#include <sstream>    // For creating text strings
#include <iomanip>    // For formatting text (std::setprecision)
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp> // For cv::ORB and cv::BFMatcher
#include "orb.hpp"                // my ORB class header
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// functions for converting between Python types and C++ types
static cv::Mat numpy_uint8_or_float_to_mat(const py::array& arr) {
    py::buffer_info info = arr.request();
    // ensure input matrix is either 2d (greyscale) or 3d (color)
    if (!(info.ndim == 2 || (info.ndim == 3 && info.shape[2] == 3)))
        throw std::runtime_error("Expect HxW (grey) or HxWx3 (BGR) image");
    if (info.format != py::format_descriptor<uint8_t>::format() &&
        info.format != py::format_descriptor<float>::format())
        throw std::runtime_error("Image must be uint8 or float32");

    int h = (int)info.shape[0], w = (int)info.shape[1];
    int c = info.ndim == 3 ? 3 : 1; // number of channels, determines opencv type

    int type;
    if (info.format == py::format_descriptor<uint8_t>::format())
        type = (c == 3) ? CV_8UC3 : CV_8UC1; // opencv's uint8 for 1 or 3 channels
    else
        type = (c == 3) ? CV_32FC3 : CV_32FC1; // opencv's float32 for 1 or 3 channels

    // make a copy of the matrix
    cv::Mat mat(h, w, type);
    // copy input buffer to cv2::Mat output
    std::memcpy(mat.data, info.ptr, (size_t)h * w * c * (type == CV_8UC1 || type == CV_8UC3 ? 1 : 4));
    return mat;
}

static py::array keypoints_to_numpy(const std::vector<cv::KeyPoint>& kps) {
    // Columns: x, y, size, angle, response, octave, class_id  (float32)
    int num_fields = 7; // number of fields of a keypoint
    py::array_t<float> out({(py::ssize_t)kps.size(), (py::ssize_t)num_fields});
    auto buf = out.mutable_unchecked<2>(); // mutable unchecked is used to access a numpy array
    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        const auto& kp = kps[(size_t)i];
        buf(i,0)=kp.pt.x; 
        buf(i,1)=kp.pt.y; 
        buf(i,2)=kp.size;  
        buf(i,3)=kp.angle;
        buf(i,4)=kp.response; 
        buf(i,5)=(float)kp.octave; 
        buf(i,6)=(float)kp.class_id;
    }
    return out;
}

static py::array mat_to_numpy_copy(const cv::Mat& m) {
    // Returns numpy array with correct dtype
    std::vector<ssize_t> shape = {(ssize_t)m.rows, (ssize_t)m.cols};
    if (m.channels() == 1) {
        if (m.depth() == CV_8U) {
            py::array_t<uint8_t> a(shape);
            // Copy input matrix to output (dest is mutable data of numpy arr)
            std::memcpy(a.mutable_data(), m.data, (size_t)m.total() * m.elemSize());
            return a;
        } else if (m.depth() == CV_32F) {
            py::array_t<float> a(shape);
            // Copy input matrix to output (dest is mutable data of numpy arr)
            std::memcpy(a.mutable_data(), m.data, (size_t)m.total() * m.elemSize());
            return a;
        }
    } else {
        // should have only one channel, error if now
        throw std::runtime_error("Unexpected multi-channel matrix for descriptors");
    }
    throw std::runtime_error("Unsupported descriptor depth (use CV_8U or CV_32F)");
}

// actual interface functions (called from Python)
py::tuple extract(const py::array& image) {
    cv::Mat frame = numpy_uint8_or_float_to_mat(image);
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    // Get one common set of keypoints using my FAST stub
    std::vector<cv::KeyPoint> common_keypoints = ORBDescriptor::detectKeypoints(gray_frame);

    // get keypoints (own implementation)
    std::vector<cv::KeyPoint> my_keypoints = common_keypoints;
    cv::Mat my_descriptors;
    if (!my_keypoints.empty())
    {
        ORBDescriptor::computeOrientation(gray_frame, my_keypoints);
        my_descriptors = ORBDescriptor::computeDescriptors(gray_frame, my_keypoints);
    }
    // it's easiest to pass np arrays via Pybind, so we convert our data to those
    py::array kp_arr = keypoints_to_numpy(my_keypoints);
    py::array desc_arr = mat_to_numpy_copy(my_descriptors);

    return py::make_tuple(kp_arr,desc_arr);
}

// Defines functions to be accessible via Python
PYBIND11_MODULE(orb_project, m) {
    m.doc() = "Feature extraction via ORB implementation";

    m.def("extract", &extract,
          py::arg("image"),
          "Extract keypoints (Nx7 float32) and descriptors (NxD uint8/float32) from image frame");
}


int main()
{
    // --- 1. INITIALIZE ---
    std::cout << "Starting live ORB verification..." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;

    // My implementation
    ORBDescriptor my_orb;

    // OpenCV's implementation (for comparison)
    cv::Ptr<cv::ORB> cv_orb = cv::ORB::create();

    // Brute-Force Matcher (do this once)
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // --- 2. START VIDEO LOOP ---
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

        // --- 3. DETECT KEYPOINTS ---
        // Get one common set of keypoints using my FAST stub
        std::vector<cv::KeyPoint> common_keypoints = my_orb.detectKeypoints(gray_frame);
        
        // --- 4. RUN BOTH IMPLEMENTATIONS ---
        
        // A) my Implementation
        std::vector<cv::KeyPoint> my_keypoints = common_keypoints;
        cv::Mat my_descriptors;
        if (!my_keypoints.empty())
        {
            my_orb.computeOrientation(gray_frame, my_keypoints);
            my_descriptors = my_orb.computeDescriptors(gray_frame, my_keypoints);
        }

        // B) OpenCV's Implementation
        std::vector<cv::KeyPoint> cv_keypoints = common_keypoints;
        cv::Mat cv_descriptors;
        if (!cv_keypoints.empty())
        {
            cv_orb->compute(gray_frame, cv_keypoints, cv_descriptors);
        }

        // --- 5. REAL-TIME MATCHING ---
        int num_good_matches = 0;
        float match_rate = 0.0f;

        if (my_descriptors.empty() || cv_descriptors.empty())
        {
            // Can't match if one set is empty (e.g., dark frame)
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
        
        // --- 6. VISUALIZE ---
        
        // A) Draw my keypoints (with my orientation)
        cv::Mat output_frame;
        cv::drawKeypoints(frame, my_keypoints, output_frame, cv::Scalar(0, 255, 0), 
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // B) Draw the verification text
        std::string kp_text = "Keypoints: " + std::to_string(common_keypoints.size());
        std::string match_text = "Good Matches: " + std::to_string(num_good_matches);
        
        std::ostringstream ss;
        ss << "Match Rate: " << std::fixed << std::setprecision(2) << match_rate << "%";
        std::string rate_text = ss.str();

        cv::putText(output_frame, kp_text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(output_frame, match_text, cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(output_frame, rate_text, cv::Point(10, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // C) Show the frame
        cv::imshow("Live ORB Verification (my Code vs OpenCV)", output_frame);

        // --- 7. QUIT ---
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    // --- 8. CLEAN UP ---
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Webcam feed stopped." << std::endl;

    return 0;
}