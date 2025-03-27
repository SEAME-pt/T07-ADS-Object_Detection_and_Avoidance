#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <SDL2/SDL.h>
#include <functional>
#include <unordered_map>
#include <array>
#include <memory>
#include "LaneDetector.hpp"
#include "JetCar.hpp"


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

enum Mode {
    MODE_JOYSTICK,
    MODE_AUTONOMOUS
};

struct Actions {
    std::function<void()> onPress;
    std::function<void()> onRelease;
};

class Controller {
public:
    Controller(JetCar* jetCar);
    ~Controller();

    void setButtonAction(int button, Actions actions);
    void setAxisAction(int axis, std::function<void(int)> action);
    void processEvent(const SDL_Event& event);
    void setMode(const int &mode);
    int  getMode();
    void listen();
    void setLaneDetector(std::unique_ptr<LaneDetector> detector);

private:
    SDL_Joystick* joystick;
    JetCar* jetCar;
    std::unique_ptr<LaneDetector> laneDetector;
    std::unordered_map<int, Actions> buttonActions;
    std::unordered_map<int, std::function<void(int)>> axisActions;
    std::array<bool, 12> buttonStates;
    int _currentMode;
    cv::Mat frame, output_frame;
    cv::VideoWriter video_writer;

    void autonomous();
};

#endif