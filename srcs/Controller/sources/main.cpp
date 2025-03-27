#include <iostream>
#include "JetCar.hpp"
#include "LaneDetector.hpp"
#include <csignal>
#include "Controller.hpp"

JetCar jetCar(0x60, 0x40);

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0);
    exit(signum);
}

void handleMotors(int value) {
    value *= -1;
    int motorSpeed = static_cast<int>((value / 32768.0) * 100);
    if (motorSpeed >= 30)
        motorSpeed = 30;
    motorSpeed = std::max(-100, std::min(100, motorSpeed));
    std::cout << "Velocidade do motor: " << motorSpeed << std::endl;
    jetCar.set_motor_speed(motorSpeed);
}

int changeMode(int mode, Controller &controller, JetCar &jetCar) {
    controller.setMode(mode == MODE_JOYSTICK ? MODE_AUTONOMOUS : MODE_JOYSTICK);
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    auto laneDetector = std::make_unique<LaneDetector>(modelPath); // Criar com unique_ptr

    if (!laneDetector->initialize()) {
        std::cerr << "Erro ao inicializar o detector de faixas!" << std::endl;
        return -1;
    }

    std::cout << "Sistema iniciado com sucesso! Pressione 'q' para sair." << std::endl;
    signal(SIGINT, signalHandler);

    try {
        Controller controller(&jetCar); // Passar ponteiro para JetCar
        controller.setLaneDetector(std::move(laneDetector)); // Transferir posse
        
        changeModeActions.onPress = nullptr;
        changeModeActions.onRelease = [&](){
            changeMode(controller.getMode(), controller, jetCar);
        };

        controller.setAxisAction(3, handleMotors);
        controller.setButtonAction(BTN_HOME, changeModeActions);


        controller.listen();
    } catch (const std::runtime_error& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}