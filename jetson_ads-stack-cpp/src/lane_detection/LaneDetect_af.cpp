#pragma once
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <cmath>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << msg << std::endl;
    }
};

class LaneDetector {

public:
    LaneDetector(const std::string& trt_model_path);
    ~LaneDetector();
    bool initialize();
    void processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask = true);
    void loadEngine(const std::string& trt_model_path);
    void preprocess(const cv::Mat& frame);
    void infer();

    //void calculateSteeringParams(int left_edge, int right_edge, int& lane_center, float& offset, float& angle);
    //void calculateLaneGeometry(float& offset, float& angle, cv::Mat& debug_img);
	bool calculateLaneGeometry(float& offset, float& angle, cv::Mat* debug_img = nullptr);
    // Helper functions
    void defineROI(int& start_y, int& end_y, int& start_x, int& end_x) const;
    bool findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi,
                       std::vector<cv::Point>& left_edges,
                       std::vector<cv::Point>& right_edges) const;
    void weightedLinearRegression(const std::vector<cv::Point>& points,
                                  double& slope, double& intercept) const;
    void calculateOffsetAndAngle(double left_slope, double left_intercept,
                                 double right_slope, double right_intercept,
                                 int y_bottom, float& offset, float& angle) const;
    void applyKalmanFilter(float measured_offset, float measured_angle,
                           float& smoothed_offset, float& smoothed_angle);
    void drawDebugInfo(cv::Mat* debug_img, const std::vector<cv::Point>& left_edges,
                       const std::vector<cv::Point>& right_edges,
                       float offset, float angle) const;
	double calculateDistance(int pixel_y, int x_length) const;

	// TensorRT
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Logger logger_;
    void* buffers_[2];
    cudaStream_t stream_;
    std::vector<float> input_data_;
    std::vector<float> output_data_;

    // OpenCV
    cv::VideoCapture cap_;
    cv::cuda::GpuMat gpu_frame_;
    cv::cuda::GpuMat gpu_resized_;
    cv::Mat lane_mask_;           // Binary lane mask (assumed to be set elsewhere)

    // Kalman Filter
    cv::KalmanFilter kf_;         // Kalman filter for smoothing offset and angle
    // cv::Mat measurement_;
    // cv::Mat prediction_;
    float offset_kalman_;
    float angle_kalman_;

    // Dimensões
    int image_width_;             // Image width, set during initialization
    int image_height_;            // Image height, set during initialization
    int frame_width_ = 640;
    int frame_height_ = 360;
    int roi_start_y_;
    int roi_end_y_;

    // Valores
    float estimated_lane_width_;
    int prev_left_edge_;
    int prev_right_edge_;

	// private:
	// Member variables

    // Fixed parameters as constants
    static constexpr double CAMERA_TILT = 0.296706; // 17 degrees in radians (17 * pi/180)
    static constexpr double CAMERA_HEIGHT = 0.15;   // 15 cm in meters
    static constexpr double METER_PER_PIXEL = 0.0005556;  // Example scale factor, should be calibrated [m/pixel]
    static constexpr float ROI_START_Y_PERCENT = 0.7f; // ROI starts at 70% of image height
    static constexpr float ROI_END_Y_PERCENT = 1.0f;   // ROI ends at 100% of image height
    static constexpr int MAX_SEARCH_DISTANCE = 500;    // Max distance (pixels) to search for edges
	static constexpr double A_DISTANCE = -2.62e-6; // Coefficient for distance calculation
	static constexpr double B_DISTANCE = 1.4722e-3;   // Coefficient for distance calculation

};

LaneDetector::LaneDetector() {

	cudaStreamCreate(&stream_);

	// Assume lane_mask_, image_width_, and image_height_ are set elsewhere

	// Initialize Kalman filter (2D state: offset, angle)
    kf_ = cv::KalmanFilter(2, 2, 0, CV_32F);
    kf_.statePre.at<float>(0) = 0.0f; // Initial offset
    kf_.statePre.at<float>(1) = 0.0f; // Initial angle
    kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1); // Identity matrix
    kf_.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1); // Identity matrix
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPre, cv::Scalar::all(1));

    input_height_ = 128;
    input_width_ = 256;

    frame_height_ = 128;
    frame_width_ = 256;
    roi_start_y_ = 0;
    // roi_start_y_ = frame_height_ / 2;
    roi_end_y_ = frame_height_ - 10;

    offset_kalman_ = 0.0f;
    angle_kalman_ = 0.0f;
    estimated_lane_width_ = 200.0f;
    prev_left_edge_ = frame_width_ / 2;
    prev_right_edge_ = frame_width_ / 2;

    loadEngine(trt_model_path);
}

LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

bool LaneDetector::initialize() {
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=360, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1";
    cap_.open(pipeline, cv::CAP_GSTREAMER);

    return cap_.isOpened();
}

bool LaneDetector::calculateLaneGeometry(float& offset, float& angle, cv::Mat* debug_img) {
    // Check if lane mask is valid
    if (lane_mask_.empty() || lane_mask_.type() != CV_32F) {
        return false;
    }

    // Step 1: Define the Region of Interest (ROI)
    int start_y, end_y, start_x, end_x;
    defineROI(start_y, end_y, start_x, end_x);
    cv::Rect roi(start_x, start_y, end_x - start_x, end_y - start_y);

    // Step 2: Find left and right lane edges using dense sampling
    std::vector<cv::Point> left_edges, right_edges;
    if (!findLaneEdges(lane_mask_, roi, left_edges, right_edges)) {
        return false; // Not enough edge points detected
    }

    // Step 3: Perform weighted linear regression to fit lines to edges
    double left_slope, left_intercept, right_slope, right_intercept;
    weightedLinearRegression(left_edges, left_slope, left_intercept);
    weightedLinearRegression(right_edges, right_slope, right_intercept);

    // Step 4: Calculate offset and angle from the fitted lines
    float measured_offset, measured_angle;
    calculateOffsetAndAngle(left_slope, left_intercept, right_slope, right_intercept,
                            end_y - 1, measured_offset, measured_angle);

    // Step 5: Apply Kalman filter to smooth the estimates
    float smoothed_offset, smoothed_angle;
    applyKalmanFilter(measured_offset, measured_angle, smoothed_offset, smoothed_angle);

    // Step 6: Set output parameters
    offset = smoothed_offset;
    angle = smoothed_angle;

    // Optional: Draw debug information if debug_img is provided
    if (debug_img != nullptr) {
        drawDebugInfo(debug_img, left_edges, right_edges, offset, angle);
    }

    return true;
}

// Define the ROI based on image dimensions and fixed percentages
void LaneDetector::defineROI(int& start_y, int& end_y, int& start_x, int& end_x) const {
    start_y = static_cast<int>(image_height_ * ROI_START_Y_PERCENT);
    end_y = static_cast<int>(image_height_ * ROI_END_Y_PERCENT);
    start_x = 0; // Full width
    end_x = image_width_;
}

// Find left and right lane edges using dense sampling
bool LaneDetector::findLaneEdges(const cv::Mat& lane_mask, const cv::Rect& roi,
                                 std::vector<cv::Point>& left_edges,
                                 std::vector<cv::Point>& right_edges) const {
    left_edges.clear();
    right_edges.clear();

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        int left_x = -1, right_x = -1;
        // Scan from left to find first edge
        for (int x = roi.x; x < roi.x + roi.width && x < roi.x + MAX_SEARCH_DISTANCE; ++x) {
            if (lane_mask.at<float>(y, x) > 0.5f) { // Threshold at 0.5 for binary [0,1]
                left_x = x;
                break;
            }
        }
        // Scan from right to find last edge
        for (int x = roi.x + roi.width - 1; x >= roi.x && x >= roi.x + roi.width - MAX_SEARCH_DISTANCE; --x) {
            if (lane_mask.at<float>(y, x) > 0.5f) {
                right_x = x;
                break;
            }
        }
        if (left_x != -1 && right_x != -1 && left_x < right_x) {
            left_edges.emplace_back(left_x, y);
            right_edges.emplace_back(right_x, y);
        }
    }
    return !left_edges.empty() && !right_edges.empty();
}

// Perform weighted linear regression (fits x = m*y + b)
void LaneDetector::weightedLinearRegression(const std::vector<cv::Point>& points,
                                            double& slope, double& intercept) const {
    if (points.size() < 2) {
        slope = 0.0;
        intercept = image_width_ / 2.0; // Default to center
        return;
    }

    double sum_w = 0.0, sum_wy = 0.0, sum_wx = 0.0, sum_wyy = 0.0, sum_wyx = 0.0;
    int start_y = static_cast<int>(image_height_ * ROI_START_Y_PERCENT);
    int end_y = static_cast<int>(image_height_ * ROI_END_Y_PERCENT);
    double range_y = end_y - start_y;

    for (const auto& pt : points) {
        double y = pt.y;
        double x = pt.x;
        double weight = (y - start_y) / range_y; // Higher weight near bottom
        weight = std::max(0.1, weight); // Minimum weight to avoid zero

        sum_w += weight;
        sum_wy += weight * y;
        sum_wx += weight * x;
        sum_wyy += weight * y * y;
        sum_wyx += weight * y * x;
    }

    double mean_y = sum_wy / sum_w;
    double mean_x = sum_wx / sum_w;
    double denom = sum_wyy - 2 * mean_y * sum_wy + sum_w * mean_y * mean_y;
    if (std::abs(denom) < 1e-6) {
        slope = 0.0;
        intercept = mean_x;
    } else {
        slope = (sum_wyx - mean_y * sum_wx - mean_x * sum_wy + sum_w * mean_x * mean_y) / denom;
        intercept = mean_x - slope * mean_y;
    }
}

// Calculate offset (meters) and angle (radians) from fitted lines
void LaneDetector::calculateOffsetAndAngle(double left_slope, double left_intercept,
                                           double right_slope, double right_intercept,
                                           int y_bottom, float& offset, float& angle) const {
    // Calculate x positions at bottom row
    double x_left = left_slope * y_bottom + left_intercept;
    double x_right = right_slope * y_bottom + right_intercept;
    double x_mid = (x_left + x_right) / 2.0;
    double x_center = image_width_ / 2.0;

    // Offset in pixels, then convert to meters
    float offset_pixels = static_cast<float>(x_mid - x_center);
    offset = offset_pixels * METER_PER_PIXEL;

    // Average slope for angle, adjust for image coordinates and camera tilt
    double avg_slope = (left_slope + right_slope) / 2.0;
    float angle_image = std::atan(avg_slope); // Positive slope = rightward in image
    angle = angle_image - static_cast<float>(CAMERA_TILT); // Adjust for downward tilt
}

// Apply Kalman filter to smooth offset and angle
void LaneDetector::applyKalmanFilter(float measured_offset, float measured_angle,
                                     float& smoothed_offset, float& smoothed_angle) {
    // Predict
    cv::Mat prediction = kf_.predict();

    // Update with measurement
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measured_offset, measured_angle);
    cv::Mat corrected = kf_.correct(measurement);

    // Extract smoothed values
    smoothed_offset = corrected.at<float>(0);
    smoothed_angle = corrected.at<float>(1);
}

// Draw debug information on the provided image
void LaneDetector::drawDebugInfo(cv::Mat* debug_img,
                                 const std::vector<cv::Point>& left_edges,
                                 const std::vector<cv::Point>& right_edges,
                                 float offset, float angle) const {
    if (debug_img->empty()) {
        *debug_img = cv::Mat(image_height_, image_width_, CV_8UC3, cv::Scalar(0));
    }
    if (debug_img->type() != CV_8UC3) {
        debug_img->convertTo(*debug_img, CV_8UC3);
    }

    // Draw edge points
    for (const auto& pt : left_edges) {
        cv::circle(*debug_img, pt, 2, cv::Scalar(0, 0, 255), -1); // Red for left
    }
    for (const auto& pt : right_edges) {
        cv::circle(*debug_img, pt, 2, cv::Scalar(0, 255, 0), -1); // Green for right
    }

    // Draw lane center line based on offset and angle (simplified)
    int y_bottom = static_cast<int>(image_height_ * ROI_END_Y_PERCENT) - 1;
    int x_center = image_width_ / 2;
    int x_mid = x_center + static_cast<int>(offset / METER_PER_PIXEL);
    cv::line(*debug_img, cv::Point(x_mid, y_bottom),
             cv::Point(x_mid - static_cast<int>(100 * std::tan(angle)), y_bottom - 100),
             cv::Scalar(255, 255, 0), 2); // Yellow line
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

    cudaMalloc(&buffers_[0], 1 * 3 * input_height_ * input_width_ * sizeof(float));
    cudaMalloc(&buffers_[1], 1 * 1 * input_height_ * input_width_ * sizeof(float));
    input_data_.resize(1 * 3 * input_height_ * input_width_);
    output_data_.resize(1 * 1 * input_height_ * input_width_);
}

void LaneDetector::infer() {
	cudaMemcpyAsync(buffers_[0], input_data_.data(), input_data_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
	context_->enqueueV2(buffers_, stream_, nullptr);
	cudaMemcpyAsync(output_data_.data(), buffers_[1], output_data_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
	cudaStreamSynchronize(stream_);
}

void LaneDetector::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_)); // resize to input size

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // Normaliza para [0,1]

    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data_.data() + c * input_height_ * input_width_, channels[c].data, input_height_ * input_width_ * sizeof(float));
    }
}

void LaneDetector::processFrame(cv::Mat& frame, float& offset, float& angle, cv::Mat& output_frame, bool visualize_mask) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());

    // Sigmoid para converter logits em probabilidades
    cv::exp(-lane_mask_, lane_mask_);
    lane_mask_ = 1.0 / (1.0 + lane_mask_);

    // Aplica threshold final
    lane_mask_ = (lane_mask_ > 0.3);

    // Resize da máscara para o tamanho do frame original
    cv::Mat resized_mask;
    cv::resize(lane_mask_, resized_mask, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_NEAREST);
    lane_mask_ = resized_mask;

    // Prepare output
    output_frame = frame.clone();

    // Calcular geometria da pista
    calculateLaneGeometry(offset, angle, output_frame);

    // Visualização do centro estimado da pista
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2;
    int lane_center = frame_width_ / 2 + static_cast<int>(offset);

    // Linha vermelha no centro da pista estimada
    cv::line(output_frame, cv::Point(lane_center, roi_mid_y), cv::Point(lane_center, roi_mid_y - 30), cv::Scalar(0, 0, 255), 2);

    // Linha azul no centro da imagem (referência)
    cv::line(output_frame, cv::Point(frame_width_ / 2, roi_mid_y), cv::Point(frame_width_ / 2, roi_mid_y - 40), cv::Scalar(255, 0, 0), 2);

    // Texto do offset e ângulo
    std::string offset_text = "Offset: " + std::to_string(static_cast<int>(offset));
    std::string angle_text = "Angle: " + std::to_string(static_cast<int>(angle)) + " deg";
    cv::putText(output_frame, offset_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(output_frame, angle_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Mostrar máscara pós-threshold no canto inferior direito
    if (visualize_mask) {
        cv::Mat threshold_mask_display;
        lane_mask_.convertTo(threshold_mask_display, CV_8U, 255.0);
        cv::resize(threshold_mask_display, threshold_mask_display, cv::Size(frame_width_ / 4, frame_height_ / 4));
        cv::cvtColor(threshold_mask_display, threshold_mask_display, cv::COLOR_GRAY2BGR);
        threshold_mask_display.copyTo(output_frame(cv::Rect(frame_width_ - threshold_mask_display.cols, frame_height_ - threshold_mask_display.rows, threshold_mask_display.cols, threshold_mask_display.rows)));
    }

    // Print no terminal
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[LaneDetector] Offset: " << offset
              << ", Angle: " << angle
              << " deg | Frame Size: " << frame.cols << "x" << frame.rows
              << std::endl;
}

double LaneDetector::calculateDistance(int pixel_y, int x_length) {
	double scale_factor = A_DISTANCE * pixel_y + B_DISTANCE;
	return scale_factor * x_length;
}
