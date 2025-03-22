#include "LaneDetection.hpp"

LaneDetector::LaneDetector(std::string& modelPath) : angle(0), offset(0) {
    engine = loadEngine(modelPath);
    if (!engine) {
        std::cerr << "Erro ao carregar o modelo TensorRT!" << std::endl;
        exit(-1);
    }

    context = engine->createExecutionContext();
    inputIndex = engine->getBindingIndex("input");
    outputIndex = engine->getBindingIndex("output");
    inputSize = 3 * 160 * 240;
    outputSize = 160 * 240;

    inputHost = new float[inputSize];
    outputHost = new float[outputSize];

    cudaMalloc((void**)&inputDevice, inputSize * sizeof(float));
    cudaMalloc((void**)&outputDevice, outputSize * sizeof(float));

    cudaStreamCreate(&stream);
}

LaneDetector::~LaneDetector() {
    cudaStreamDestroy(stream);
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    delete[] inputHost;
    delete[] outputHost;
    delete context;
    delete engine;
}

nvinfer1::ICudaEngine* LaneDetector::loadEngine(std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        std::cerr << "Erro ao abrir o arquivo " << modelPath << "!" << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    return runtime->deserializeCudaEngine(buffer.data(), size);
}

std::vector<float> LaneDetector::preprocessImage(cv::Mat& img) {
    cv::Mat resized, floatImage, normalized;
    cv::resize(img, resized, cv::Size(240, 160));
    resized.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    cv::cvtColor(floatImage, normalized, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);
    std::vector<float> chwData;
    for (const auto& channel : channels) {
        chwData.insert(chwData.end(), (float*)channel.datastart, (float*)channel.dataend);
    }
    return chwData;
}

void LaneDetector::calculateOffsetAndAngle(cv::Mat& frame, cv::Mat& laneMask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(laneMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Point2f leftLaneCenter, rightLaneCenter;
    bool leftLaneFound = false, rightLaneFound = false;

    for (const auto& contour : contours) {
        cv::Moments moments = cv::moments(contour);
        cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);

        if (center.x < frame.cols / 2) {
            leftLaneCenter = center;
            leftLaneFound = true;
        } else {
            rightLaneCenter = center;
            rightLaneFound = true;
        }
    }

    if (leftLaneFound && rightLaneFound) {
        float laneCenterX = (leftLaneCenter.x + rightLaneCenter.x) / 2;
        offset = (frame.cols / 2) - laneCenterX;
        angle = std::atan2(rightLaneCenter.y - leftLaneCenter.y, rightLaneCenter.x - leftLaneCenter.x) * 180.0f / CV_PI;
    }
}

bool LaneDetector::runInference(cv::Mat& frame) {
    std::vector<float> preprocessed = preprocessImage(frame);
    std::memcpy(inputHost, preprocessed.data(), inputSize * sizeof(float));

    cudaMemcpyAsync(inputDevice, inputHost, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    void* bindings[] = {inputDevice, outputDevice};
    if (!context->enqueueV2(bindings, stream, nullptr)) {
        std::cerr << "Erro ao executar inferência!" << std::endl;
        return false;
    }

    cudaMemcpyAsync(outputHost, outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat laneMask(160, 240, CV_32F, outputHost);
    laneMask = (laneMask > 0.3);
    laneMask.convertTo(laneMask, CV_8U);
    laneMask *= 255;

    cv::Mat laneMaskResized;
    cv::resize(laneMask, laneMaskResized, cv::Size(frame.cols, frame.rows));

    calculateOffsetAndAngle(frame, laneMaskResized);

    return true;
}

float LaneDetector::getAngle() const {
    return angle;
}

float LaneDetector::getOffset() const {
    return offset;
}

void LaneDetector::setAngle(float a) {
    angle = a;
}

void LaneDetector::setOffset(float o) {
    offset = o;
}
