#include "Controller.hpp"
#include <iostream>

Controller::Controller(JetCar* jetCar) : joystick(nullptr), jetCar(jetCar) {
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 Joystick: " + std::string(SDL_GetError()));
    }

    int joystickCount = SDL_NumJoysticks();
    std::cout << "Número de joysticks conectados: " << joystickCount << std::endl;

    _currentMode = MODE_JOYSTICK;

    buttonStates[10] = false;
    buttonStates[11] = false;

    if (joystickCount > 0) {
        joystick = SDL_JoystickOpen(0);
        if (joystick) {
            std::cout << "Joystick 0 conectado!" << std::endl;
        } else {
            throw std::runtime_error("Falha ao abrir o joystick: " + std::string(SDL_GetError()));
        }
    } else {
        throw std::runtime_error("Nenhum joystick detectado.");
    }

    std::string pipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
                          "rtph264pay ! udpsink host=0.0.0.0 port=5000";
    video_writer.open(pipeline, cv::CAP_GSTREAMER, 0, 30.0, cv::Size(640, 360), true);
    if (!video_writer.isOpened()) {
        throw std::runtime_error("Falha ao abrir o VideoWriter para streaming!");
    }
    std::cout << "Streaming iniciado em udp://0.0.0.0:5000" << std::endl;
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

void Controller::setAxisAction(int axis, std::function<void(int)> action) {
    axisActions[axis] = action;
}

void Controller::processEvent(const SDL_Event& event) {
    if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) {
        bool isPressed = (event.type == SDL_JOYBUTTONDOWN);
        int button = event.jbutton.button;

        if (button < static_cast<int>(buttonStates.size())) { // Corrigir comparação signed/unsigned
            std::cout << "Button " << button << " " << (isPressed ? "pressed" : "released") << std::endl;
            buttonStates[button] = isPressed;
            if (buttonActions.find(button) != buttonActions.end()) {
                if (isPressed) {
                    if (buttonActions[button].onPress)
                        buttonActions[button].onPress();
                } else {
                    if (buttonActions[button].onRelease)
                        buttonActions[button].onRelease();
                }
            }
        }
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode != MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;

        std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    else if (event.type == SDL_JOYAXISMOTION && _currentMode == MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;

        std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end() && axis == 3) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYDEVICEADDED) {
        std::cout << "Joystick on!" << std::endl;

        if (!joystick) {
            joystick = SDL_JoystickOpen(0);
            if (joystick) {
                std::cout << "Joystick 0 conectado!" << std::endl;
            } else {
                throw std::runtime_error("Falha ao abrir o joystick: " + std::string(SDL_GetError()));
            }
        }
    } else if (event.type == SDL_JOYDEVICEREMOVED) {
        std::cout << "Joystick off!" << std::endl;

        if (joystick) {
            SDL_JoystickClose(joystick);
            joystick = nullptr;
        }
        exit(1);
    }
}

void Controller::setMode(const int &mode) {
    _currentMode = mode;
}

int Controller::getMode() {
    return _currentMode;
}

void Controller::listen() {
    SDL_Event event;
    while (true) {
        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }

        if (_currentMode == MODE_AUTONOMOUS) {
            std::cout << "Modo autônomo ativado!" << std::endl;
            autonomous();
        }

        if (buttonStates[10] && buttonStates[11]) {
            break;
        }

        if (!joystick) {
            std::cout << "No joystick conected, quiting..." << std::endl;
            break;
        }

        SDL_Delay(10);
    }
}

void Controller::autonomous() {

    if (!laneDetector || !laneDetector->cap_.read(frame)) {
        std::cerr << "🚨 Erro: Não foi possível capturar a imagem ou LaneDetector não inicializado!" << std::endl;
        return;
    }

    laneDetector->processFrame(frame, output_frame);

    // Usar offset_kalman e angle_kalman (valores filtrados)
    float angle = laneDetector->angle_kalman;
    float offset = laneDetector->offset_kalman;

    std::cout << "Ângulo: " << angle << " graus" << std::endl;
    std::cout << "Offset: " << offset << " pixels" << std::endl;

    // cv::imshow("Lane Detection", output_frame);
    video_writer.write(output_frame);

    float steering = std::clamp(angle * 3, -90.0f, 90.0f);
    std::cout << "Steering: " << steering << std::endl;
    jetCar->set_servo_angle(steering);


    if (cv::waitKey(1) == 'q') {
        jetCar->set_servo_angle(0);
        jetCar->set_motor_speed(0);
        exit(0);
    }
}

void Controller::setLaneDetector(std::unique_ptr<LaneDetector> detector) {
    laneDetector = std::move(detector); // Transferir posse do ponteiro
}