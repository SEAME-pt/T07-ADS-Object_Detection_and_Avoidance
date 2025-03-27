#include "LaneDetector.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

LaneDetector::LaneDetector(const std::string& trt_model_path) {
    cudaStreamCreate(&stream_);

    kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
    kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);
    kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1);
    cv::setIdentity(kalman_.processNoiseCov, cv::Scalar::all(0.03));
    cv::setIdentity(kalman_.measurementNoiseCov, cv::Scalar::all(1.0));
    cv::setIdentity(kalman_.errorCovPost, cv::Scalar::all(1.0));
    measurement_ = cv::Mat(2, 1, CV_32F);
    prediction_ = cv::Mat(4, 1, CV_32F);

    loadEngine(trt_model_path);
    offset_kalman = 0.0f;
    angle_kalman = 0.0f;
    roi_start_y_ = 120;
    roi_end_y_ = 240;
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
    if (!cap_.isOpened()) {
        std::cerr << "🚨 Erro: Não foi possível acessar a câmera!" << std::endl;
        return false;
    }
    return true;
}

void LaneDetector::loadEngine(const std::string& trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Erro ao abrir o modelo TensorRT!" << std::endl;
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

void LaneDetector::preprocess(const cv::Mat& frame) {
    gpu_frame_.upload(frame);
    cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(input_width_, input_height_));
    cv::Mat resized;
    gpu_resized_.download(resized);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
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

    // Analisar a região central
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
        // Relaxar a condição: aceitar bordas mesmo se apenas uma for encontrada
        if (temp_left != center_x) left_edges.push_back(temp_left);
        if (temp_right != center_x) right_edges.push_back(temp_right);
    }

    // Depuração: verificar se a ROI tem valores
    if (left_edges.empty() && right_edges.empty()) {
        std::cout << "Nenhuma borda detectada na ROI (" << roi_start_y_ << " a " << roi_end_y_ << ")" << std::endl;
        // Opcional: salvar a máscara para análise
        static int frame_count = 0;
        if (frame_count++ % 30 == 0) {
            cv::imwrite("debug_mask_" + std::to_string(frame_count) + ".png", mask_8u);
        }
    }

    // Média das bordas detectadas
    if (!left_edges.empty()) {
        left_edge = std::accumulate(left_edges.begin(), left_edges.end(), 0) / left_edges.size();
    }
    if (!right_edges.empty()) {
        right_edge = std::accumulate(right_edges.begin(), right_edges.end(), 0) / right_edges.size();
    }
}

void LaneDetector::calculateSteeringParams(int left_edge, int right_edge, int& lane_center, float& offset, float& angle) {
    lane_center = (left_edge + right_edge) / 2;
    int camera_center = frame_width_ / 2;
    offset = static_cast<float>(lane_center - camera_center);
    angle = atan2(offset, frame_height_) * 180.0 / CV_PI;
}

void LaneDetector::processFrame(cv::Mat& frame, cv::Mat& output_frame) {
    preprocess(frame);
    infer();

    lane_mask_ = cv::Mat(input_height_, input_width_, CV_32F, output_data_.data());
    cv::resize(lane_mask_, lane_mask_, cv::Size(frame_width_, frame_height_));
    lane_mask_ = (lane_mask_ > 0.5);

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

    output_frame = frame.clone();
    int roi_mid_y = (roi_start_y_ + roi_end_y_) / 2;
    cv::line(output_frame, cv::Point(left_edge, roi_mid_y), cv::Point(left_edge, roi_mid_y - 20), cv::Scalar(0, 255, 0), 2);
    cv::line(output_frame, cv::Point(right_edge, roi_mid_y), cv::Point(right_edge, roi_mid_y - 20), cv::Scalar(0, 255, 0), 2);
    cv::line(output_frame, cv::Point(lane_center, roi_mid_y), cv::Point(lane_center, roi_mid_y - 30), cv::Scalar(0, 0, 255), 2);
    cv::line(output_frame, cv::Point(frame_width_ / 2, roi_mid_y), cv::Point(frame_width_ / 2, roi_mid_y - 40), cv::Scalar(255, 0, 0), 2);

    char text[128];
    sprintf(text, "Offset: %.2f px", offset_kalman);
    cv::putText(output_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    sprintf(text, "Angle: %.2f deg", angle_kalman);
    cv::putText(output_frame, text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    // Desenhar limites da ROI
    cv::line(output_frame, cv::Point(0, roi_start_y_), cv::Point(frame_width_, roi_start_y_), cv::Scalar(255, 255, 0), 1);
    cv::line(output_frame, cv::Point(0, roi_end_y_), cv::Point(frame_width_, roi_end_y_), cv::Scalar(255, 255, 0), 1);
}
