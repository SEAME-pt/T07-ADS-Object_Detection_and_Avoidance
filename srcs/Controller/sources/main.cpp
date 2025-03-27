#include <iostream>
#include "JetCar.hpp"
#include "LaneDetector.hpp"
#include <csignal>

JetCar jetCar(0x60, 0x40);

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0);
    exit(signum);
}

void moveForwardandBackward(int value) {
    
    value -= 16319;

    value = (value / 165) * -1;
    
    std::cout << "Axis moved to " << value << std::endl;
    car.setMotorSpeed(value);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    LaneDetector laneDetector(modelPath);

    if (!laneDetector.initialize()) {
        std::cerr << "Erro ao inicializar o detector de faixas!" << std::endl;
        return -1;
    }

    controller.setLaneDetector(laneDetector);

    std::cout << "Sistema iniciado com sucesso! Pressione 'q' para sair." << std::endl;
    signal(SIGINT, signalHandler);

    cv::Mat frame, output_frame;

    Controller controller;
    controller.setAxisAction(5, moveForwardandBackward);
    controller.listen();

    return 0;
}

