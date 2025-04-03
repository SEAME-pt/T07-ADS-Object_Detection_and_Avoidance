#include "LaneDetector.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_);

    // Initialize Kalman filter
    kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
    kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1);
    cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(0.03));
    cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(1.0));
    cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.0));
    measurement_ = cv::Mat(2, 1, CV_32F);
    prediction_ = cv::Mat(4, 1, CV_32F);

    // Model input dimensions
    input_height_ = 128;
    input_width_ = 256;

    // Frame dimensions from camera
    frame_height_ = 360;
    frame_width_ = 640;

    // ROI set to the middle third of the frame
    roi_start_y_ = frame_height_ / 3;     // 120 (1/3 of 360)
    roi_end_y_ = (frame_height_ * 2) / 3; // 240 (2/3 of 360)

    loadEngine(trt_model_path);
    offset_kalman = 0.0f;
    angle_kalman = 0.0f;
}

LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

bool LaneDetector::initialize() {
    // GStreamer pipeline for camera capture (640x360)
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
        std::cerr << "🚨 Error: Could not access the camera!" << std::endl;
        return false;
    }
    return true;
}

void LaneDetector::loadEngine(const std::string& trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening TensorRT model file!" << std::endl;
        return;
    }

    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(trt_model.data(), trt_model.size(), nullptr));
    context_.reset(engine_->createExecutionContext());

    // Allocate CUDA buffers for input and output
    cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float)); // Input: 3 channels
    cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float)); // Output: 1 channel
    input_data_.resize(1 * 3 * input_height_ * input_width_);
    output_data_.resize(1 * 1 * input_height_ * input_width_);
}

void LaneDetector::preprocess(const cv::Mat& frame) {
    // Crop the frame to the ROI (y=120 to y=240, 640x120)
    cv::Rect roi(0, roi_start_y_, frame_width_, roi_end_y_ - roi_start_y_); // 0, 120, 640, 120
    cv::Mat cropped_frame = frame(roi);

    // The cropped frame is 640x120, which has an aspect ratio of 640/120 ≈ 5.33
    // The model expects 256x128 (aspect ratio 2:1), so we need to resize while preserving the aspect ratio
    float model_aspect = static_cast<float>(input_width_) / input_height_; // 2.0
    float crop_aspect = static_cast<float>(cropped_frame.cols) / cropped_frame.rows; // 640/120 ≈ 5.33

    int resize_width, resize_height;
    if (crop_aspect > model_aspect) {
        // Crop is wider than model aspect: fit width, adjust height
        resize_width = input_width_; // 256
        resize_height = static_cast<int>(input_width_ / crop_aspect); // 256 / (640/120) ≈ 48
    } else {
        // Crop is taller than model aspect: fit height, adjust width
        resize_height = input_height_; // 128
        resize_width = static_cast<int>(input_height_ * crop_aspect); // 128 * (640/120) ≈ 682
    }

    // Resize to intermediate size (e.g., 256x48)
    gpu_frame_.upload(cropped_frame);
    cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(resize_width, resize_height));

    // Pad to the exact model input size (256x128)
    cv::Mat resized;
    gpu_resized_.download(resized);
    cv::Mat padded = cv::Mat::zeros(input_height_, input_width_, resized.type()); // 128x256
    int pad_top = (input_height_ - resize_height) / 2; // (128 - 48) / 2 = 40
    int pad_left = (input_width_ - resize_width) / 2; // (256 - 256) / 2 = 0
    cv::Rect pad_roi(pad_left, pad_top, resize_width, resize_height); // 0, 40, 256, 48
    resized.copyTo(padded(pad_roi));

    // Normalize and split channels
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(padded, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

void LaneDetector::infer() {
    cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->enqueueV2(buffers_, stream_, nullptr);
    cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void LaneDetector::findLaneEdges(int& left_edge, int& right_edge) {
    cv::Mat mask_8u;
    lane_mask_.convertTo(mask_8u, CV_8U, 255.0);
    int center_x = frame_width_ / 2;
    left_edge = center_x;
    right_edge = center_x;

    std::vector<int> left_edges, right_edges;
    for (int y = roi_start_y_; y < roi_end_y_; ++y) {
        uchar* row = mask_8u.ptr<uchar>(y);
        int temp_left = center_x;
        int temp_right = center_x;

        for (int x = center_x; x >= 0; --x) {
            if (row[x] > 0) {
                temp_left = x;
                break;
            }
        }
        for (int x = center_x; x < frame_width_; ++x) {
            if (row[x] > 0) {
                temp_right = x;
                break;
            }
        }
        if (temp_left != center_x) left_edges.push_back(temp_left);
        if (temp_right != center_x) right_edges.push_back(temp_right);
    }

    if (left_edges.empty() && right_edges.empty()) {
        std::cout << "No edges detected in ROI (" << roi_start_y_ << " to " << roi_end_y_ << ")" << std::endl;
        static int frame_count = 0;
        if (frame_count++ % 30 == 0) {
            cv::imwrite("debug_mask_" + std::to_string(frame_count) + ".png", mask_8u);
            std::cout << "Saving debug mask: debug_mask_" << frame_count << ".png" << std::endl;
        }
    }

    if (!left_edges.empty()) {
        left_edge = std::accumulate(left_edges.begin(), left_edges.end(), 0) / left_edges.size();
    }
    if (!right_edges.empty()) {
        right_edge = std::accumulate(right_edges.begin(), right_edges.end(), 0) / right_edges.size();
    }

    std::cout << "Left Edge: " << left_edge << ", Right Edge: " << right_edge << std::endl;
}

void LaneDetector::calculateSteeringParams(int left_edge, int right_edge, int& lane_center, float& offset, float& angle) {
    const int lane_width = 200; // Estimated lane width in pixels
    const int desired_offset = 100; // Desired distance from detected line to car center
    int camera_center = frame_width_ / 2;
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2;

    if (left_edge != camera_center && right_edge != camera_center) {
        lane_center = (left_edge + right_edge) / 2;
        std::cout << "Both lines detected, lane_center: " << lane_center << std::endl;
    }
    else if (left_edge != camera_center && right_edge == camera_center) {
        // Right curve: keep left line at desired_offset pixels left of center
        lane_center = left_edge + desired_offset;
        std::cout << "Only left line detected, adjusting lane_center to fixed distance: " << lane_center << std::endl;
    }
    else if (left_edge == camera_center && right_edge != camera_center) {
        // Left curve: keep right line at desired_offset pixels right of center
        lane_center = right_edge - desired_offset;
        std::cout << "Only right line detected, adjusting lane_center to fixed distance: " << lane_center << std::endl;
    }
    else {
        lane_center = camera_center + static_cast<int>(offset_kalman);
        std::cout << "No edges detected, using offset_kalman: " << offset_kalman << std::endl;
    }

    lane_center = std::max(0, std::min(lane_center, frame_width_ - 1));

    offset = static_cast<float>(lane_center - camera_center);
    angle = atan2(offset, frame_height_ - roi_mid_y) * 180.0 / CV_PI;

    if (offset > frame_width_ / 2) offset = frame_width_ / 2;
    if (offset < -frame_width_ / 2) offset = -frame_width_ / 2;
    if (angle > 90.0f) angle = 90.0f;
    if (angle < -90.0f) angle = -90.0f;

    std::cout << "Lane Center: " << lane_center << ", Offset: " << offset << ", Angle: " << angle << std::endl;
}

void LaneDetector::processFrame(cv::Mat& frame, cv::Mat& output_frame, bool visualize_mask = true) {
    // Save the input frame for debugging
    static int frame_count = 0;
    if (frame_count++ % 30 == 0) {
        std::string filename = "input_frame_" + std::to_string(frame_count) + ".png";
        cv::imwrite(filename, frame);
        std::cout << "Saved input frame: " << filename << std::endl;
    }

    preprocess(frame);
    infer();

    // Generate the lane mask from raw output
    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data()); // 128x256

    // Debug: Print min and max values of the raw output
    double min_val, max_val;
    cv::minMaxLoc(lane_mask_, &min_val, &max_val);
    std::cout << "Raw output min: " << min_val << ", max: " << max_val << std::endl;

    // Apply sigmoid to convert logits to probabilities
    cv::exp(-lane_mask_, lane_mask_);
    lane_mask_ = 1.0 / (1.0 + lane_mask_);

    // Debug: Print min and max values after sigmoid
    cv::minMaxLoc(lane_mask_, &min_val, &max_val);
    std::cout << "After sigmoid min: " << min_val << ", max: " << max_val << std::endl;

    // Resize the mask to the ROI size (640x120)
    int roi_height = roi_end_y_ - roi_start_y_; // 240 - 120 = 120
    float model_aspect = static_cast<float>(input_width_) / input_height_; // 2.0
    float roi_aspect = static_cast<float>(frame_width_) / roi_height; // 640/120 ≈ 5.33

    int resize_width, resize_height;
    if (roi_aspect > model_aspect) {
        resize_width = frame_width_; // 640
        resize_height = static_cast<int>(frame_width_ / model_aspect); // 640 / 2 = 320
    } else {
        resize_height = roi_height; // 120
        resize_width = static_cast<int>(roi_height * model_aspect);
    }

    // Resize to intermediate size (e.g., 640x320)
    cv::resize(lane_mask_, lane_mask_, cv::Size(resize_width, resize_height)); // 640x320

    // Crop or pad to the exact ROI size (640x120)
    cv::Mat resized_mask;
    if (resize_height > roi_height) {
        // Crop the resized mask to 640x120
        int crop_top = (resize_height - roi_height) / 2; // (320 - 120) / 2 = 100
        cv::Rect crop_roi(0, crop_top, frame_width_, roi_height); // 0, 100, 640, 120
        resized_mask = lane_mask_(crop_roi);
    } else {
        // Pad the resized mask to 640x120
        resized_mask = cv::Mat::zeros(roi_height, frame_width_, lane_mask_.type());
        int pad_top = (roi_height - resize_height) / 2;
        cv::Rect pad_roi(0, pad_top, frame_width_, resize_height);
        lane_mask_.copyTo(resized_mask(pad_roi));
    }

    // Place the resized mask into a full-size mask (640x360)
    cv::Mat full_mask = cv::Mat::zeros(frame_height_, frame_width_, lane_mask_.type()); // 360x640
    cv::Rect roi(0, roi_start_y_, frame_width_, roi_end_y_ - roi_start_y_); // 0, 120, 640, 120
    resized_mask.copyTo(full_mask(roi));

    lane_mask_ = full_mask;
    lane_mask_ = (lane_mask_ > 0.5);

    // Visualize the binary mask if enabled
    if (visualize_mask) {
        cv::Mat mask_display;
        lane_mask_.convertTo(mask_display, CV_8U, 255.0);
        //cv::imshow("Binary Lane Mask", mask_display);
        //video_writer.write(mask_display);
        cv::waitKey(1);

        if (frame_count % 30 == 0) {
            std::string filename = "lane_mask_" + std::to_string(frame_count) + ".png";
            cv::imwrite(filename, mask_display);
            std::cout << "Saved binary mask: " << filename << std::endl;
        }
    }

    int left_edge, right_edge;
    findLaneEdges(left_edge, right_edge);
    int lane_center;

    calculateSteeringParams(left_edge, right_edge, lane_center, offset, angle);

    measurement_.at<float>(0) = offset;
    measurement_.at<float>(1) = angle;
    kalman_.correct(measurement_);
    prediction_ = kalman_.predict();
    offset_kalman = prediction_.at<float>(0);
    angle_kalman = prediction_.at<float>(2);

    if (offset_kalman > frame_width_ / 2) offset_kalman = frame_width_ / 2;
    if (offset_kalman < -frame_width_ / 2) offset_kalman = -frame_width_ / 2;
    if (angle_kalman > 90.0f) angle_kalman = 90.0f;
    if (angle_kalman < -90.0f) angle_kalman = -90.0f;

    output_frame = frame.clone();
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2; // (120 + 240) / 2 = 180
    cv::line(output_frame, cv::Point(left_edge, roi_mid_y), cv::Point(left_edge, roi_mid_y - 20), cv::Scalar(0, 255, 0), 2);
    cv::line(output_frame, cv::Point(right_edge, roi_mid_y), cv::Point(right_edge, roi_mid_y - 20), cv::Scalar(0, 255, 0), 2);
    cv::line(output_frame, cv::Point(lane_center, roi_mid_y), cv::Point(lane_center, roi_mid_y - 30), cv::Scalar(0, 0, 255), 2);
    cv::line(output_frame, cv::Point(frame_width_ / 2, roi_mid_y), cv::Point(frame_width_ / 2, roi_mid_y - 40), cv::Scalar(255, 0, 0), 2);

    char text[128];
    sprintf(text, "Offset: %.2f px", offset_kalman);
    cv::putText(output_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    sprintf(text, "Angle: %.2f deg", angle_kalman);
    cv::putText(output_frame, text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    // Draw ROI boundaries
    cv::line(output_frame, cv::Point(0, roi_start_y_), cv::Point(frame_width_, roi_start_y_), cv::Scalar(255, 255, 0), 1);
    cv::line(output_frame, cv::Point(0, roi_end_y_), cv::Point(frame_width_, roi_end_y_), cv::Scalar(255, 255, 0), 1);
}
