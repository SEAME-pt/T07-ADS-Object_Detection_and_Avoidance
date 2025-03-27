#include <iostream>
#include "JetCar.hpp"
#include "LaneDetection.hpp"
#include <csignal>

JetCar jetCar(0x60, 0x40);


void signalHandler( int signum ) {
   std::cout << "Interrupt signal (" << signum << ") received.\n";

   jetCar.set_servo_angle(0);
   jetCar.set_motor_speed(0);

   exit(signum);  
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " path to model file" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    LaneDetector laneDetector(modelPath);

    // Pipeline para captura da câmera
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=240, height=160, "
                           "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                           "videoconvert ! video/x-raw, format=BGR ! appsink";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    std::cout << "Camera aberta com sucesso!" << std::endl;

    signal(SIGINT, signalHandler); 

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Erro ao capturar o frame!" << std::endl;
            break;
        }

        if (laneDetector.runInference(frame)) {
            float angle = laneDetector.getAngle();
            float offset = laneDetector.getOffset();

            std::cout << "Ângulo: " << angle << " graus" << std::endl;
            std::cout << "Offset: " << offset << " pixels" << std::endl;

            // Exibir texto na imagem
            cv::putText(frame, "Angle: " + std::to_string(angle) + " degrees", cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Offset: " + std::to_string(offset) + " pixels", cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // Máscara da pista
            cv::Mat laneMask(160, 240, CV_32F, laneDetector.outputHost);
            laneMask = (laneMask > 0.3);
            laneMask.convertTo(laneMask, CV_8U);
            laneMask *= 255;

            cv::Mat laneMaskResized;
            cv::resize(laneMask, laneMaskResized, frame.size(), 0, 0, cv::INTER_NEAREST);
            cv::Mat laneMaskResizedColor;
            cv::cvtColor(laneMaskResized, laneMaskResizedColor, cv::COLOR_GRAY2BGR);

            cv::Mat frameWithMask;
            cv::addWeighted(frame, 1.0, laneMaskResizedColor, 0.5, 0, frameWithMask);
            cv::imshow("Lane Detection", frameWithMask);

            float steering = angle * 0.05f;
            float speed = std::max(50.0f - std::abs(offset) * 0.5f, 20.0f);

	    std::cout << "Angulo: " << angle << std::endl;

            jetCar.set_servo_angle(steering);
            jetCar.set_motor_speed(speed);
        }

        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}

