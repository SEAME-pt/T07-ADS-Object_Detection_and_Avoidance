#include <Controller.hpp>
#include <iostream>
#include "DebugTools.hpp"

Controller::Controller(JetCar* jetCar) : joystick(nullptr), jetCar(jetCar), _currentMode("") {
    // Initialize SDL for joystick input
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 Joystick: " + std::string(SDL_GetError()));
    }

    int joystickCount = SDL_NumJoysticks();
    std::cout << "Number of joysticks connected: " << joystickCount << std::endl;

    buttonStates.fill(false);  // Initialize all button states to false

    if (joystickCount > 0) {
        joystick = SDL_JoystickOpen(0);
        if (joystick) {
            std::cout << "Joystick 0 connected!" << std::endl;
        } else {
            throw std::runtime_error("Failed to open joystick: " + std::string(SDL_GetError()));
        }
    } else {
        throw std::runtime_error("No joystick detected.");
    }
}

Controller::~Controller() {
    if (joystick) {
        SDL_JoystickClose(joystick);
    }
    SDL_Quit();
}

void Controller::setButtonAction(int button, Actions actions) {
    buttonActions[button] = actions;
}

void Controller::setCurrentMode(const std::string &mode) {
    _currentMode = mode;
}

void Controller::setAxisAction(int axis, std::function<void(int)> action) {
    axisActions[axis] = action;
}

void Controller::processEvent(const SDL_Event& event) {
    if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) {
        bool isPressed = (event.type == SDL_JOYBUTTONDOWN);
        int button = event.jbutton.button;

        if (button < static_cast<int>(buttonStates.size())) {
            std::cout << "Button " << button << " " << (isPressed ? "pressed" : "released") << std::endl;
            buttonStates[button] = isPressed;
            if (buttonActions.find(button) != buttonActions.end()) {
                if (isPressed && buttonActions[button].onPress) {
                    buttonActions[button].onPress();
                } else if (!isPressed && buttonActions[button].onRelease) {
                    buttonActions[button].onRelease();
                }
            }
        }
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode == "") {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode == "") {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end() && axis == 3) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYDEVICEADDED) {
        std::cout << "Joystick connected!" << std::endl;
        if (!joystick) {
            joystick = SDL_JoystickOpen(0);
            if (joystick) {
                std::cout << "Joystick 0 connected!" << std::endl;
            } else {
                throw std::runtime_error("Failed to open joystick: " + std::string(SDL_GetError()));
            }
        }
    } else if (event.type == SDL_JOYDEVICEREMOVED) {
        std::cout << "Joystick disconnected!" << std::endl;
        if (joystick) {
            SDL_JoystickClose(joystick);
            joystick = nullptr;
        }
        exit(1);
    }
}

std::string Controller::getMode() {
    return _currentMode;
}

void Controller::listen() {
    SDL_Event event;
    while (true) {
        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }

        if (_currentMode == "acceleration_measurement") {
            DebugTools::measureAccelerationTime(*jetCar, 4);
            _currentMode = "";
        }

        if (_currentMode == "custom_pwm_value_speed") {
            int customPwmValue = 40;
            jetCar->set_motor_speed(customPwmValue);
            _currentMode = "";
        }

        if (buttonStates[BTN_SELECT] && buttonStates[BTN_START]) {
            break;
        }

        if (!joystick) {
            std::cout << "No joystick connected, quitting..." << std::endl;
            break;
        }

        SDL_Delay(10);
    }
}
