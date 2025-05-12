#include <iostream>
#include "JetCar.hpp"
#include <csignal>
#include "Controller.hpp"
#include "DebugTools.hpp"

JetCar jetCar(0x60, 0x40);

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0);
    exit(signum);
}

void handleSteering(int value) {
    int servoAngle = static_cast<int>((value / 32768.0) * 90);
    servoAngle = std::max(-90, std::min(90, servoAngle));
    jetCar.set_servo_angle(servoAngle);
}

void handleMotors(int value) {
    value *= -1;
    int motorSpeed = static_cast<int>((value / 32768.0) * 100);
    jetCar.set_motor_speed(motorSpeed);
}

int main(int argc, char *argv[]) {

    std::cout << "Sistema iniciado com sucesso! Pressione 'q' para sair." << std::endl;
    signal(SIGINT, signalHandler);

    Controller controller(&jetCar);
    try {
        
        Actions activateAcellerationMeasurement;
        activateAcellerationMeasurement.onPress = [&controller]() {
            controller.setCurrentMode("acceleration_measurement");
        };

        Actions sendCustomPwmValueSpeed;
        sendCustomPwmValueSpeed.onPress = [&controller]() {
            controller.setCurrentMode("custom_pwm_value_speed");
        };

        Actions pidAutoTune;
        pidAutoTune.onPress = []() {
            DebugTools::autoTunePID(jetCar);
        };
        
        controller.setButtonAction(BTN_B, sendCustomPwmValueSpeed);
        controller.setButtonAction(BTN_A, activateAcellerationMeasurement);
        controller.setButtonAction(BTN_X, pidAutoTune);

        controller.setAxisAction(3, handleMotors);
        controller.setAxisAction(0, handleSteering);

        controller.listen();
    } catch (const std::runtime_error& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}