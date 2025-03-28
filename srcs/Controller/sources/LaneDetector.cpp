#include "LaneDetector.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp> // For cv::fitLine

/**
 * @brief Constructor for LaneDetector class.
 * @param trt_model_path Path to the TensorRT model file.
 */
LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_); // Create CUDA stream for asynchronous operations

    // Initialize Kalman filter with 4 states (position, velocity) and 2 measurements
    kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
    kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0); // Measurement matrix H
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1); // State transition matrix F
    cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(0.03)); // Process noise covariance Q
    cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(1.0)); // Measurement noise covariance R
    cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.0)); // Posteriori error covariance P
    measurement_ = cv::Mat(2, 1, CV_32F); // Measurement vector
    prediction_ = cv::Mat(4, 1, CV_32F); // Prediction vector

    loadEngine(trt_model_path); // Load TensorRT engine
    offset_kalman = 0.0f; // Initialize Kalman offset
    angle_kalman = 0.0f; // Initialize Kalman angle
    roi_start_y_ = 360 / 2; // Central third starting y-coordinate
    roi_end_y_ = 350; // Central third ending y-coordinate
}

/**
 * @brief Destructor for LaneDetector class.
 */
LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream_); // Destroy CUDA stream
    cudaFree(buffers_[0]); // Free input buffer memory
    cudaFree(buffers_[1]); // Free output buffer memory
}

/**
 * @brief Initializes the camera capture pipeline.
 * @return True if initialization succeeds, false otherwise.
 */
bool LaneDetector::initialize() {
    // GStreamer pipeline for camera capture
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
                          "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                          "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER); // Open camera with pipeline
    if (!cap_.isOpened()) {
        std::cerr << "🚨 Error: Could not access the camera!" << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Loads the TensorRT engine from a file.
 * @param trt_model_path Path to the TensorRT model file.
 */
void LaneDetector::loadEngine(const std::string& trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary); // Open model file in binary mode
    if (!file.good()) {
        std::cerr << "Error opening TensorRT model!" << std::endl;
        return;
    }

    // Read entire file into vector
    std::vector<char> trt_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_)); // Create runtime
    engine_.reset(runtime_->deserializeCudaEngine(trt_model.data(), trt_model.size(), nullptr)); // Deserialize engine
    context_.reset(engine_->createExecutionContext()); // Create execution context

    // Allocate CUDA memory for input and output buffers
    cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float));
    cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float));
    input_data_.resize(1 * 3 * input_height_ * input_width_); // Resize input data vector
    output_data_.resize(1 * 1 * input_height_ * input_width_); // Resize output data vector
}

/**
 * @brief Preprocesses the input frame for inference.
 * @param frame Input frame to preprocess.
 */
void LaneDetector::preprocess(const cv::Mat& frame) {
    gpu_frame_.upload(frame); // Upload frame to GPU
    cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(input_width_, input_height_)); // Resize on GPU
    cv::Mat resized;
    gpu_resized_.download(resized); // Download resized frame
    resized.convertTo(resized, CV_32F, 1.0 / 255.0); // Convert to float and normalize
    std::vector<cv::Mat> channels;
    cv::split(resized, channels); // Split into RGB channels
    // Copy each channel to input data
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

/**
 * @brief Performs inference using the TensorRT engine.
 */
void LaneDetector::infer() {
    // Copy input data to GPU asynchronously
    cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->enqueueV2(buffers_, stream_, nullptr); // Execute inference
    // Copy output data back to host asynchronously
    cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_); // Wait for stream operations to complete
}

/**
 * @brief Finds lane points in the processed mask by searching outward from the center.
 * @param lane_points Vector to store detected lane points.
 */
void LaneDetector::findLanePoints(std::vector<cv::Point>& lane_points) {
    cv::Mat mask_8u;
    lane_mask_.convertTo(mask_8u, CV_8U, 255.0); // Convert mask to 8-bit unsigned
    int center_x = frame_width_ / 2; // Calculate frame center
    lane_points.clear(); // Clear previous points

    // Scan ROI in 10-pixel intervals vertically
    for (int y = roi_start_y_; y < roi_end_y_; y += 10) {
        uchar* row = mask_8u.ptr<uchar>(y); // Get row pointer
        std::vector<int> row_points;

        // Search right from center
        for (int x = center_x; x < frame_width_; ++x) {
            if (row[x] > 0) {
                row_points.push_back(x); // Add x-coordinate if pixel is part of lane
                break; // Stop searching right once a point is found
            }
        }

        // Search left from center
        for (int x = center_x - 1; x >= 0; --x) { // Start one pixel left of center to avoid overlap
            if (row[x] > 0) {
                row_points.push_back(x); // Add x-coordinate if pixel is part of lane
                break; // Stop searching left once a point is found
            }
        }

        // If points found, calculate the average x-coordinate
        if (!row_points.empty()) {
            int avg_x = std::accumulate(row_points.begin(), row_points.end(), 0) / row_points.size(); // Calculate average x
            lane_points.push_back(cv::Point(avg_x, y)); // Add point to vector
        }
    }

    // Debug output if no points found
    if (lane_points.empty()) {
        std::cout << "No lane points detected in ROI (" << roi_start_y_ << " to " << roi_end_y_ << ")" << std::endl;
        static int frame_count = 0;
        if (frame_count++ % 30 == 0) {
            cv::imwrite("debug_mask_" + std::to_string(frame_count) + ".png", mask_8u); // Save debug mask
            std::cout << "Saving debug mask: debug_mask_" << frame_count << ".png" << std::endl;
        }
    }
}

/**
 * @brief Calculates steering parameters based on lane points.
 * @param lane_points Detected lane points.
 * @param lane_center Calculated center of the lane.
 * @param offset Offset from camera center.
 * @param angle Steering angle in degrees.
 */
void LaneDetector::calculateSteeringParams(const std::vector<cv::Point>& lane_points, int& lane_center, float& offset, float& angle) {
    const int desired_offset = 100; // Desired distance from line to car center
    int camera_center = frame_width_ / 2; // Camera center x-coordinate
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2; // Middle of ROI

    if (lane_points.empty()) {
        lane_center = camera_center + static_cast<int>(offset_kalman); // Use Kalman prediction if no points
        std::cout << "No points detected, using offset_kalman: " << offset_kalman << std::endl;
    }
    else {
        cv::Vec4f line_params;
        cv::fitLine(lane_points, line_params, cv::DIST_L2, 0, 0.01, 0.01); // Fit line to points
        float vx = line_params[0]; // Line direction x
        float vy = line_params[1]; // Line direction y
        float x0 = line_params[2]; // Line start x
        float y0 = line_params[3]; // Line start y

        // Calculate line point at roi_mid_y
        float t = (roi_mid_y - y0) / vy;
        int line_x = static_cast<int>(x0 + t * vx);

        bool is_left_line = line_x < camera_center; // Determine line position

        // Adjust lane center based on line position
        if (is_left_line) {
            lane_center = line_x + desired_offset; // Line on left, center on right
            std::cout << "Left line detected, lane_center: " << lane_center << std::endl;
        }
        else {
            lane_center = line_x - desired_offset; // Line on right, center on left
            std::cout << "Right line detected, lane_center: " << lane_center << std::endl;
        }
    }

    lane_center = std::max(0, std::min(lane_center, frame_width_ - 1)); // Clamp lane center

    offset = static_cast<float>(lane_center - camera_center); // Calculate offset
    angle = atan2(offset, frame_height_ - roi_mid_y) * 180.0 / CV_PI; // Calculate angle in degrees

    // Clamp offset and angle values
    if (offset > frame_width_ / 2) offset = frame_width_ / 2;
    if (offset < -frame_width_ / 2) offset = -frame_width_ / 2;
    if (angle > 90.0f) angle = 90.0f;
    if (angle < -90.0f) angle = -90.0f;

    std::cout << "Lane Center: " << lane_center << ", Offset: " << offset << ", Angle: " << angle << std::endl;
}

/**
 * @brief Processes a frame and generates output with lane detection.
 * @param frame Input frame to process.
 * @param output_frame Output frame with visualizations.
 */
void LaneDetector::processFrame(cv::Mat& frame, cv::Mat& output_frame) {
    preprocess(frame); // Preprocess frame
    infer(); // Run inference

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data()); // Create mask from output
    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_)); // Resize to frame size
    lane_mask_ = (lane_mask_ > 0.5); // Threshold mask

    std::vector<cv::Point> lane_points;
    findLanePoints(lane_points); // Find lane points
    int lane_center;

    calculateSteeringParams(lane_points, lane_center, offset, angle); // Calculate steering parameters

    // Update Kalman filter
    measurement_.at<float>(0) = offset;
    measurement_.at<float>(1) = angle;
    kalman_.correct(measurement_); // Correct with measurement
    prediction_ = kalman_.predict(); // Predict next state
    offset_kalman = prediction_.at<float>(0); // Update Kalman offset
    angle_kalman = prediction_.at<float>(2); // Update Kalman angle

    // Clamp Kalman values
    if (offset_kalman > frame_width_ / 2) offset_kalman = frame_width_ / 2;
    if (offset_kalman < -frame_width_ / 2) offset_kalman = -frame_width_ / 2;
    if (angle_kalman > 90.0f) angle_kalman = 90.0f;
    if (angle_kalman < -90.0f) angle_kalman = -90.0f;

    output_frame = frame.clone(); // Clone input frame for output
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2; // Middle of ROI

    // Draw detected points and lines
    for (const auto& point : lane_points) {
        cv::circle(output_frame, point, 3, cv::Scalar(0, 255, 255), -1); // Yellow points
    }
    cv::line(output_frame, cv::Point(lane_center, roi_mid_y), cv::Point(lane_center, roi_mid_y - 30), cv::Scalar(0, 0, 255), 2); // Red lane center line
    cv::line(output_frame, cv::Point(frame_width_ / 2, roi_mid_y), cv::Point(frame_width_ / 2, roi_mid_y - 40), cv::Scalar(255, 0, 0), 2); // Blue camera center line

    // Add text overlays
    char text[128];
    sprintf(text, "Offset: %.2f px", offset_kalman);
    cv::putText(output_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2); // Green offset text
    sprintf(text, "Angle: %.2f deg", angle_kalman);
    cv::putText(output_frame, text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2); // Green angle text

    // Draw ROI boundaries
    cv::line(output_frame, cv::Point(0, roi_start_y_), cv::Point(frame_width_, roi_start_y_), cv::Scalar(255, 255, 0), 1); // Cyan start line
    cv::line(output_frame, cv::Point(0, roi_end_y_), cv::Point(frame_width_, roi_end_y_), cv::Scalar(255, 255, 0), 1); // Cyan end line
}

/**
 * @brief Gets the current offset from the Kalman filter.
 * @return Offset in pixels.
 */
float LaneDetector::getOffset() const {
    return offset_kalman;
}

/**
 * @brief Gets the current angle from the Kalman filter.
 * @return Angle in degrees.
 */
float LaneDetector::getAngle() const {
    return angle_kalman;
}