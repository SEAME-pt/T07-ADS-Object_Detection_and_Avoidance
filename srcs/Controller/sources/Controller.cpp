#include "Controller.hpp"

Controller::Controller() : gameController(nullptr) {
    if (SDL_Init(SDL_INIT_GAMECONTROLLER) < 0) {
        throw std::runtime_error("Failed to initialize SDL2 GameController: " + std::string(SDL_GetError()));
    }

    // Abrir o primeiro controlador de jogo disponível
    for (int i = 0; i < SDL_NumJoysticks(); ++i) {
        if (SDL_IsGameController(i)) {
            gameController = SDL_GameControllerOpen(i);
            if (gameController) {
                std::cout << "GameController connected: " << SDL_GameControllerName(gameController) << std::endl;
                break;
            }
        }
    }

    if (!gameController) {
        throw std::runtime_error("No compatible game controller found.");
    }
}

Controller::~Controller() {
    if (gameController) {
        SDL_GameControllerClose(gameController);
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
    if (event.type == SDL_CONTROLLERBUTTONDOWN || event.type == SDL_CONTROLLERBUTTONUP) {
        bool isPressed = (event.type == SDL_CONTROLLERBUTTONDOWN);
        int button = event.cbutton.button;

        if (button < buttonStates.size()) {
            buttonStates[button] = isPressed;
            std::cout << "button " << button << std::endl;
            if (buttonActions.find(button) != buttonActions.end()) {
                if (isPressed) {
                    if(buttonActions[button].onPress)
                        buttonActions[button].onPress();
                } else {
                    if(buttonActions[button].onRelease)
                        buttonActions[button].onRelease();
                }
            }
        }
    } else if (event.type == SDL_CONTROLLERAXISMOTION) {
        int axis = event.caxis.axis;
        int value = event.caxis.value;

        // std::cout << "Axis " << axis << " moved to " << value << "." << std::endl;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_CONTROLLERDEVICEREMOVED) {
        // Handle the disconnection of the game controller
        std::cout << "Game controller disconnected." << std::endl;

        // Close the controller and set to nullptr
        if (gameController) {
            SDL_GameControllerClose(gameController);
            gameController = nullptr;
        }

        std::cout << "Exiting the program due to controller disconnection." << std::endl;
        exit(1);
    }
}

void Controller::listen() {
    SDL_Event event;
    while (true) {
        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }
        if (buttonStates[4] && buttonStates[6]) {
            break;
        }
        // If no controller is connected, break the loop or handle accordingly
        if (!gameController) {
            std::cout << "No controller connected, stopping listen." << std::endl;
            break;
        }

        if (!gameController) {
            std::cout << "No controller connected, stopping listen." << std::endl;
            break;
        }
        SDL_Delay(10);
    }
}