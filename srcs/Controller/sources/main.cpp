#include <iostream>
#include "JetCar.hpp"
#include "Controller.hpp"

JetCar jetCar(0x60, 0x40);

void handleSteering(int value) {
    int servoAngle = static_cast<int>((value / 32768.0) * 90);
    servoAngle = std::max(-90, std::min(90, servoAngle));
    jetCar.set_servo_angle(servoAngle);
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
}

int main() {

    Actions changeModeActions;

    try {
        Controller controller(&jetCar);
        
        changeModeActions.onPress = nullptr;
        changeModeActions.onRelease = [&](){
            changeMode(controller.getMode(), controller, jetCar);
        };


        jetCar.set_servo_angle(0);
        jetCar.set_motor_speed(0); 

        controller.setAxisAction(0, handleSteering);
        controller.setAxisAction(3, handleMotors);
        controller.setButtonAction(BTN_HOME, changeModeActions);

        controller.listen();
    } catch (const std::exception& e) {
        jetCar.set_servo_angle(0);
        jetCar.set_motor_speed(0); 
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }
    jetCar.set_servo_angle(0);
    jetCar.set_motor_speed(0); 

    return 0;
}
