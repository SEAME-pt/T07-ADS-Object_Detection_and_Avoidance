#pragma once

#include <iostream>
#include <functional>
#include <array>
#include <map>
#include <SDL2/SDL.h>
#include <zmq.hpp>

struct Actions {
    std::function<void()> onPress;
    std::function<void()> onRelease;
};

class Controller {
public:
    Controller();
    ~Controller();
    
    void setButtonAction(int button, Actions actions);
    void setAxisAction(int axis, std::function<void(int)> action);
    void listen();

private:
    SDL_GameController* gameController;
    std::map<int, Actions> buttonActions;
    std::map<int, std::function<void(int)>> axisActions; // Mapeia eixos a ações que recebem um valor
    std::array<bool, SDL_CONTROLLER_BUTTON_MAX> buttonStates; // Armazenar estados dos botões
    void processEvent(const SDL_Event& event);
};