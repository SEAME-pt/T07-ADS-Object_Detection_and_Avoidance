#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Logger para TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class LaneDetector {
public:
    LaneDetector(std::string& modelPath);
    ~LaneDetector();

    bool runInference(cv::Mat& frame);
    float getAngle() const;
    float getOffset() const;
    void setAngle(float angle);
    void setOffset(float offset);

private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    Logger logger;
    float angle;
    float offset;

    // TensorRT
    float* inputHost;
    float* outputHost;
    float* inputDevice;
    float* outputDevice;
    int inputSize;
    int outputSize;
    int inputIndex;
    int outputIndex;
    cudaStream_t stream;

    // Métodos auxiliares
    nvinfer1::ICudaEngine* loadEngine(std::string& modelPath);
    std::vector<float> preprocessImage(cv::Mat& img);
    void calculateOffsetAndAngle(cv::Mat& frame, cv::Mat& laneMask);
};

#endif // LANE_DETECTOR_HPP
