#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << msg << std::endl;
    }
};

class LaneDetector {
public:
    /**
     * @brief Constructor for LaneDetector class.
     * @param trt_model_path Path to the TensorRT model file.
     */
    LaneDetector(const std::string& trt_model_path);

    /**
     * @brief Destructor for LaneDetector class.
     */
    ~LaneDetector();

    /**
     * @brief Initializes the camera capture pipeline.
     * @return True if initialization succeeds, false otherwise.
     */
    bool initialize();

    /**
     * @brief Processes a frame and generates output with lane detection.
     * @param frame Input frame to process.
     * @param output_frame Output frame with visualizations.
     */
    void processFrame(cv::Mat& frame, cv::Mat& output_frame);

    /**
     * @brief Gets the current offset from the Kalman filter.
     * @return Offset in pixels.
     */
    float getOffset() const;

    /**
     * @brief Gets the current angle from the Kalman filter.
     * @return Angle in degrees.
     */
    float getAngle() const;

    cv::VideoCapture cap_;

private:
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
    cv::cuda::GpuMat gpu_frame_;
    cv::cuda::GpuMat gpu_resized_;
    cv::Mat lane_mask_;

    // Kalman Filter
    cv::KalmanFilter kalman_;
    cv::Mat measurement_;
    cv::Mat prediction_;

    // Dimensions
    int input_width_ = 240;
    int input_height_ = 160;
    int frame_width_ = 640;
    int frame_height_ = 360;

    // Values
    float offset_kalman;
    float angle_kalman;
    int roi_start_y_;
    int roi_end_y_;
    float angle;
    float offset;

    /**
     * @brief Loads the TensorRT engine from a file.
     * @param trt_model_path Path to the TensorRT model file.
     */
    void loadEngine(const std::string& trt_model_path);

    /**
     * @brief Preprocesses the input frame for inference.
     * @param frame Input frame to preprocess.
     */
    void preprocess(const cv::Mat& frame);

    /**
     * @brief Performs inference using the TensorRT engine.
     */
    void infer();

    /**
     * @brief Finds lane points in the processed mask by searching outward from the center.
     * @param lane_points Vector to store detected lane points.
     */
    void findLanePoints(std::vector<cv::Point>& lane_points);

    /**
     * @brief Calculates steering parameters based on lane points.
     * @param lane_points Detected lane points.
     * @param lane_center Calculated center of the lane.
     * @param offset Offset from camera center.
     * @param angle Steering angle in degrees.
     */
    void calculateSteeringParams(const std::vector<cv::Point>& lane_points, int& lane_center, float& offset, float& angle);
};

#endif // LANE_DETECTOR_HPP
