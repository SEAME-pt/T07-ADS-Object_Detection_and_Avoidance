#pragma once

#include <iostream>
#include <functional>
#include <array>
#include <map>
#include <SDL2/SDL.h>
#include <zmq.hpp>
#include "JetCar.hpp"
#include "LaneDetector.hpp"

#define BTN_A 0
#define BTN_B 1
#define BTN_X 3
#define BTN_Y 4
#define BTN_LB 6
#define BTN_RB 7
#define BTN_SELECT 10
#define BTN_START 11
#define BTN_HOME 12
#define BTN_LSTICK 13
#define BTN_RSTICK 14

struct Actions {
    std::function<void()> onPress;
    std::function<void()> onRelease;
};

enum Mode {
    MODE_JOYSTICK,
    MODE_AUTONOMOUS
};

class Controller {
public:
    Controller(JetCar* jetCar);
    ~Controller();
    
    void setButtonAction(int button, Actions actions);
    void setAxisAction(int axis, std::function<void(int)> action);
    void setMode(int const &mode);
    void autonomous();

    int     getMode();

    void    listen();

private:
    SDL_Joystick*   joystick;
    JetCar*         jetCar;
    int             _currentMode;
    LaneDetector    laneDetector;

    std::map<int, Actions> buttonActions;
    std::map<int, std::function<void(int)>> axisActions;
    std::array<bool, SDL_CONTROLLER_BUTTON_MAX> buttonStates;
    void processEvent(const SDL_Event& event);

    void setLaneDetector(LaneDetector laneDetector);
};