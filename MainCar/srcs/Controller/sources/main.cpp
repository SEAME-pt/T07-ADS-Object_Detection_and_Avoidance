#include "Controller.hpp"
#include "JetCar.hpp"
#include "devices.hpp"
#include <iostream>
#include <pigpio.h>
#include <zmq.h>

JetCar car;

void moveLeftandRight(int value) {
     
    float fvalue = value * 1.0;// + 6900;
  //  fvalue -= -129.0;
    std::cout << "Before: " << fvalue << std::endl;
    fvalue = fvalue / 32000.0 * 45;
    std::cout << "Axis moved to " << fvalue << std::endl;
    car.setServoAngle(fvalue);
}

void moveForwardandBackward(int value) {
    
    value -= 16319;

    value = (value / 165) * -1;
    
    std::cout << "Axis moved to " << value << std::endl;
    car.setMotorSpeed(value);
}

int main() {
    std::cout << "Team07 Controller App" << std::endl;
    zmq::context_t context(1);
    zmq::socket_t publisher_controls(context, zmq::socket_type::pub);
    publisher_controls.bind("tcp://*:5556");

    // Actions horn;
    // horn.onPress = [&publisher_controls]() {
    //     hornOnPressed(publisher_controls);
    // };
    // horn.onRelease = [&publisher_controls]() {
    //     hornOnReleased(publisher_controls);
    // };

    Actions car_break;
    car_break.onPress = [&publisher_controls]() {
        breakOnPressed(publisher_controls);
    };
    car_break.onRelease = [&publisher_controls]() {
        breakOnReleased(publisher_controls);
    };

    // For toggle actions just use the onPress
    Actions lowLights;
    lowLights.onPress = [&publisher_controls]() {
        lightsLowToggle(publisher_controls);
    };

    Actions highLights;
    highLights.onPress = [&publisher_controls]() {
        lightsHighToggle(publisher_controls);
    };

    Actions leftLights;
    leftLights.onPress = [&publisher_controls]() {
        indicationLightsLeft(publisher_controls);
    };

    Actions rightLights;
    rightLights.onPress = [&publisher_controls]() {
        indicationLightsRight(publisher_controls);
    };

    Actions emergencyLights;
    emergencyLights.onPress = [&publisher_controls]() {
        emergencyOnLights(publisher_controls);
    };


    try {
        Controller controller;
        // Start listening for gamepad events

        controller.setButtonAction(BTN_A, car_break);
        controller.setButtonAction(BTN_X, emergencyLights);

        controller.setButtonAction(BTN_L, rightLights);
        controller.setButtonAction(BTN_R, leftLights);
        
        controller.setButtonAction(BTN_B, lowLights);
        controller.setButtonAction(BTN_Y, highLights);


        controller.setAxisAction(SDL_CONTROLLER_AXIS_LEFTX, moveLeftandRight);
        controller.setAxisAction(5, moveForwardandBackward);

        controller.listen();

    } catch (const std::runtime_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}