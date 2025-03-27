#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include <SDL2/SDL.h>
#include <functional>
#include <unordered_map>
#include <array>
#include <memory>
#include "LaneDetector.hpp"
#include "JetCar.hpp"

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
    int getMode();
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