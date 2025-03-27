#include "Controller.hpp"

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
    if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) { // Joypad button
        bool isPressed = (event.type == SDL_JOYBUTTONDOWN);
        int button = event.jbutton.button;

        if (button < buttonStates.size()) {
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
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode != MODE_AUTONOMOUS) { // Joypad axis motion
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;

        std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYDEVICEREMOVED) { // Joypad removed
        std::cout << "Joystick off!" << std::endl;

        if (joystick) {
            SDL_JoystickClose(joystick);
            joystick = nullptr;
        }
        exit(1);
    }
}

void Controller::setMode(int const &mode) {
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
           autonomous();
        }

        if (buttonStates[10] && buttonStates[11]) { // Start + Select exit program
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

    if (!laneDetector.cap_.read(frame)) {
        std::cerr << "🚨 Erro: Não foi possível capturar a imagem!" << std::endl;
        return ;
    }

    laneDetector.processFrame(frame, output_frame);

    float angle = laneDetector.angle; 
    float offset = laneDetector.offset;

    std::cout << "Ângulo: " << angle << " graus" << std::endl;
    std::cout << "Offset: " << offset << " pixels" << std::endl;

    cv::imshow("Lane Detection", output_frame);

    float steering = std::clamp(angle * 3, -90.0f, 90.0f);
    std::cout << "Angulo: " << angle << std::endl;
    jetCar->set_servo_angle(steering);

}
