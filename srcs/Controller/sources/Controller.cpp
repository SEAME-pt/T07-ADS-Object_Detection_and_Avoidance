#include <Controller.hpp>
#include <iostream>

Controller::Controller(JetCar* jetCar) : joystick(nullptr), jetCar(jetCar), _currentMode(MODE_JOYSTICK) {
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

    // Setup video streaming pipeline
    std::string pipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
                          "rtph264pay ! udpsink host=192.168.43.190 port=5000 sync=false";
    video_writer.open(pipeline, cv::CAP_GSTREAMER, 0, 30.0, cv::Size(640, 360), true);
    if (!video_writer.isOpened()) {
        throw std::runtime_error("Failed to open VideoWriter for streaming!");
    }
    std::cout << "Streaming started at udp://0.0.0.0:5000" << std::endl;
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
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode != MODE_AUTONOMOUS) {
        int axis = event.jaxis.axis;
        int value = event.jaxis.value;
        std::cout << "Axis " << axis << " moved to " << value << std::endl;
        if (axisActions.find(axis) != axisActions.end()) {
            axisActions[axis](value);
        }
    } else if (event.type == SDL_JOYAXISMOTION && _currentMode == MODE_AUTONOMOUS) {
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
            std::cout << "Autonomous mode activated!" << std::endl;
            autonomous();
        }

        if (buttonStates[BTN_SELECT] && buttonStates[BTN_START]) {
            break;
        }

        if (!joystick) {
            std::cout << "No joystick connected, quitting..." << std::endl;
            break;
        }

        SDL_Delay(10);  // Small delay to avoid overloading CPU
    }
}

Controller::State Controller::kinematicModel(const State& state, float delta, float a) {
    // Kinematic bicycle model to predict next state
    State next;
    next.x = state.x + state.v * std::cos(state.theta) * DT;
    next.y = state.y + state.v * std::sin(state.theta) * DT;
    next.theta = state.theta + (state.v / L) * std::tan(delta) * DT;
    next.v = state.v + a * DT;
    return next;
}

void Controller::setupCostFunction(Eigen::MatrixXd& H, Eigen::VectorXd& f, const Eigen::VectorXd& y_ref, const Eigen::VectorXd& theta_ref) {
    // Initialize cost matrices
    H.setZero();
    f.setZero();

    // Populate cost function for QP problem
    for (int i = 0; i < N; ++i) {
        H(i, i) = Q_y;                // Penalize lateral offset
        H(N + i, N + i) = Q_theta;    // Penalize heading error
        H(2 * N + i, 2 * N + i) = R_delta;  // Penalize steering effort
        H(3 * N + i, 3 * N + i) = R_a;      // Penalize acceleration effort
        f(i) = -Q_y * y_ref[i];             // Linear term for offset
        f(N + i) = -Q_theta * theta_ref[i]; // Linear term for angle
    }

    // Add penalty for steering rate of change
    for (int i = 1; i < N; ++i) {
        H(2 * N + i, 2 * N + i) += R_d_delta;
        H(2 * N + i - 1, 2 * N + i - 1) += R_d_delta;
    }
}

Eigen::VectorXd Controller::solveMPC(const State& initial_state, const Eigen::VectorXd& y_ref, const Eigen::VectorXd& theta_ref) {
    int n_vars = 4 * N;  // Variables: y, theta, delta, a for each timestep
    Eigen::MatrixXd H(n_vars, n_vars);
    Eigen::VectorXd f(n_vars);

    setupCostFunction(H, f, y_ref, theta_ref);

    // Temporary proportional control until OSQP is integrated
    Eigen::VectorXd control_sequence(2);
    // Steering: Use offset and average predicted angle for curve anticipation
    float steering = (0.5f * y_ref[0]) + (2.0f * theta_ref.mean());
    control_sequence[0] = steering;

    // Throttle: Base speed adjusted by predicted curvature
    float base_throttle = 0.8f;
    float throttle = base_throttle - 0.05f * fabs(theta_ref.mean());
    control_sequence[1] = throttle;

    return control_sequence;
}

void Controller::autonomous() {
    static float prev_angle = 0.0f;  // Store previous angle for rate calculation

    // Check if LaneDetector is initialized and capture frame
    if (!laneDetector || !laneDetector->cap_.read(frame)) {
        std::cerr << "Error: Could not capture frame or LaneDetector not initialized!" << std::endl;
        return;
    }

    float offset, angle;
    tracker.mark();
    laneDetector->processFrame(frame, offset, angle, output_frame, true);
    
    //tracker.mark();
    

    // Calculate rate of change of angle to predict curve
    float angle_rate = (angle - prev_angle) / DT;  // deg/s
    prev_angle = angle;

    // Convert to MPC inputs
    float y_ref = offset * (1.0f/640.0f);  // Convert pixels to meters (adjust scale if needed)
    float theta_ref = angle * (CV_PI / 180.0f);  // Convert degrees to radians

    // Predict future trajectory over horizon
    Eigen::VectorXd y_ref_vec(N);
    Eigen::VectorXd theta_ref_vec(N);
    for (int i = 0; i < N; ++i) {
        float t = i * DT;
        y_ref_vec[i] = y_ref;  // Assume constant offset for simplicity (can add offset_rate if desired)
        theta_ref_vec[i] = theta_ref + (angle_rate * (CV_PI / 180.0f) * t);  // Linear extrapolation of angle
    }

    // Solve MPC to get control inputs
    Eigen::VectorXd control = solveMPC(current_state_, y_ref_vec, theta_ref_vec);
    float delta = control[0];  // Steering angle (radians)
    float a = control[1];      // Acceleration (m/s²)

    // Apply constraints
    float steering = std::max(-MAX_DELTA, std::min(MAX_DELTA, delta));  // Limit to ±90 deg in radians
    float throttle = std::max(0.2f, std::min(0.8f, a));  // Throttle range adjusted

    // Update vehicle state
    current_state_ = kinematicModel(current_state_, steering, throttle);

    tracker.mark();
    std::cout << "Delta: " << tracker.delta() << " microseconds" << std::endl;
    // Apply controls to JetCar
    jetCar->set_servo_angle(steering * (180.0f / CV_PI));  // Convert radians to degrees for JetCar
    jetCar->set_motor_speed(throttle * 35);  // Scale throttle to match expected range

    // Output control values
    //std::cout << "Steering: " << steering << " rad (" << (steering * 180.0f / CV_PI) << " deg), Throttle: " << throttle << std::endl;

    video_writer.write(output_frame);  // Stream the output frame
}

void Controller::setLaneDetector(std::unique_ptr<LaneDetector> detector) {
    laneDetector = std::move(detector);
}
